from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.special import fresnel
from scipy.ndimage import gaussian_filter
from PIL import Image
import base64
import io
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


class Aperture:
    def __init__(self, size_pixels=512, pixel_size=1e-7):
        self.size_pixels = size_pixels
        self.pixel_size = pixel_size
        self.physical_size = size_pixels * pixel_size
        
    def get_transmission(self):
        raise NotImplementedError
    
    def calculate_fraunhofer(self, wavelength):
        transmission = self.get_transmission()
        padded_size = self.size_pixels * 4
        padded = np.zeros((padded_size, padded_size))
        start = (padded_size - self.size_pixels) // 2
        padded[start:start+self.size_pixels, start:start+self.size_pixels] = transmission
        F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(padded)))
        intensity = np.abs(F) ** 2
        center = padded_size // 2
        half = self.size_pixels // 2
        intensity = intensity[center-half:center+half, center-half:center+half]
        return self._normalize(intensity)
        
    def calculate_fresnel(self, wavelength, distance):
        transmission = self.get_transmission()
        k = 2 * np.pi / wavelength
        fx = np.fft.fftfreq(self.size_pixels, self.pixel_size)
        fy = np.fft.fftfreq(self.size_pixels, self.pixel_size)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))
        T_fft = np.fft.fft2(transmission)
        U_fft = T_fft * H
        U = np.fft.ifft2(U_fft)
        intensity = np.abs(U) ** 2
        return self._normalize(intensity)
    
    def _normalize(self, arr):
        arr = np.abs(arr)
        max_val = np.max(arr)
        if max_val > 0:
            return arr / max_val
        return arr


class SingleSlitAperture(Aperture):
    def __init__(self, slit_width_pixels, size_pixels=512, pixel_size=1e-7):
        super().__init__(size_pixels, pixel_size)
        self.slit_width_pixels = slit_width_pixels
        self.slit_width = slit_width_pixels * pixel_size
        
    def get_transmission(self):
        aperture = np.zeros((self.size_pixels, self.size_pixels))
        center = self.size_pixels // 2
        hw = self.slit_width_pixels // 2
        left = max(0, center - hw)
        right = min(self.size_pixels, center + hw)
        aperture[:, left:right] = 1.0
        return aperture
    
    def calculate_fresnel(self, wavelength, distance):
        x = np.linspace(-self.physical_size/2, self.physical_size/2, self.size_pixels)
        y = np.linspace(-self.physical_size/2, self.physical_size/2, self.size_pixels)
        X, Y = np.meshgrid(x, y)
        scale = np.sqrt(2 / (wavelength * distance))
        u1 = (X - self.slit_width/2) * scale
        u2 = (X + self.slit_width/2) * scale
        S1, C1 = fresnel(u1)
        S2, C2 = fresnel(u2)
        delta_C = C2 - C1
        delta_S = S2 - S1
        intensity = delta_C**2 + delta_S**2
        return self._normalize(intensity)


class MultiSlitAperture(Aperture):
    def __init__(self, n_slits, slit_width_pixels, slit_separation_pixels, 
                 size_pixels=512, pixel_size=1e-7):
        super().__init__(size_pixels, pixel_size)
        self.n_slits = n_slits
        self.slit_width_pixels = slit_width_pixels
        self.slit_separation_pixels = slit_separation_pixels
        self.slit_width = slit_width_pixels * pixel_size
        self.period = slit_separation_pixels * pixel_size
        
    def get_transmission(self):
        aperture = np.zeros((self.size_pixels, self.size_pixels))
        center = self.size_pixels // 2
        hw = self.slit_width_pixels // 2
        
        if self.n_slits % 2 == 1:
            offsets = range(-(self.n_slits//2), self.n_slits//2 + 1)
        else:
            offsets = [i + 0.5 for i in range(-(self.n_slits//2), self.n_slits//2)]
        
        for offset in offsets:
            pos = int(center + offset * self.slit_separation_pixels)
            left = max(0, pos - hw)
            right = min(self.size_pixels, pos + hw)
            if left < right:
                aperture[:, left:right] = 1.0
            
        return aperture
    
    def calculate_fresnel(self, wavelength, distance):
        x = np.linspace(-self.physical_size/2, self.physical_size/2, self.size_pixels)
        y = np.linspace(-self.physical_size/2, self.physical_size/2, self.size_pixels)
        X, Y = np.meshgrid(x, y)
        k = 2 * np.pi / wavelength
        scale = np.sqrt(2 / (wavelength * distance))
        
        if self.n_slits % 2 == 1:
            slit_centers = [i * self.period for i in range(-(self.n_slits//2), self.n_slits//2 + 1)]
        else:
            slit_centers = [(i + 0.5) * self.period for i in range(-(self.n_slits//2), self.n_slits//2)]
        
        U_total = np.zeros_like(X, dtype=complex)
        
        for center in slit_centers:
            edge1 = center - self.slit_width/2
            edge2 = center + self.slit_width/2
            u1 = (X - edge1) * scale
            u2 = (X - edge2) * scale
            S1, C1 = fresnel(u1)
            S2, C2 = fresnel(u2)
            U_slit = (C2 - C1) + 1j * (S2 - S1)
            phase_factor = np.exp(-1j * k * center * X / distance)
            U_total += U_slit * phase_factor
        
        intensity = np.abs(U_total) ** 2
        return self._normalize(intensity)


class CircularAperture(Aperture):
    def __init__(self, radius_pixels, size_pixels=512, pixel_size=1e-7):
        super().__init__(size_pixels, pixel_size)
        self.radius_pixels = radius_pixels
        self.radius = radius_pixels * pixel_size
        self.diameter = 2 * self.radius
        
    def get_transmission(self):
        aperture = np.zeros((self.size_pixels, self.size_pixels))
        center = self.size_pixels // 2
        y, x = np.ogrid[:self.size_pixels, :self.size_pixels]
        mask = (x - center)**2 + (y - center)**2 <= self.radius_pixels**2
        aperture[mask] = 1.0
        return aperture


class ArbitraryAperture(Aperture):
    def __init__(self, transmission_array, pixel_size=1e-7):
        size_pixels = transmission_array.shape[0]
        super().__init__(size_pixels, pixel_size)
        self.transmission = transmission_array.astype(float)
        
    def get_transmission(self):
        return self.transmission.copy()


def create_aperture(aperture_type, **kwargs):
    size = kwargs.get('size_pixels', 512)
    pxsz = kwargs.get('pixel_size', 1e-7)
    
    if aperture_type == 'single':
        return SingleSlitAperture(
            slit_width_pixels=kwargs.get('slit_width', 20),
            size_pixels=size,
            pixel_size=pxsz
        )
    elif aperture_type == 'double':
        return MultiSlitAperture(
            n_slits=2,
            slit_width_pixels=kwargs.get('slit_width', 20),
            slit_separation_pixels=kwargs.get('slit_separation', 60),
            size_pixels=size,
            pixel_size=pxsz
        )
    elif aperture_type == 'triple':
        return MultiSlitAperture(
            n_slits=3,
            slit_width_pixels=kwargs.get('slit_width', 20),
            slit_separation_pixels=kwargs.get('slit_separation', 60),
            size_pixels=size,
            pixel_size=pxsz
        )
    elif aperture_type == 'n_slit':
        return MultiSlitAperture(
            n_slits=kwargs.get('n_slits', 5),
            slit_width_pixels=kwargs.get('slit_width', 10),
            slit_separation_pixels=kwargs.get('slit_separation', 40),
            size_pixels=size,
            pixel_size=pxsz
        )
    elif aperture_type == 'circle':
        return CircularAperture(
            radius_pixels=kwargs.get('radius', 40),
            size_pixels=size,
            pixel_size=pxsz
        )
    elif aperture_type == 'image':
        img_data = kwargs.get('image_data')
        transmission = image_to_transmission(img_data, size)
        return ArbitraryAperture(transmission, pxsz)
    elif aperture_type == 'matrix':
        mat_data = kwargs.get('matrix_data')
        transmission = matrix_to_transmission(mat_data, size)
        return ArbitraryAperture(transmission, pxsz)
    else:
        return SingleSlitAperture(20, size, pxsz)


def image_to_transmission(image_data, size):
    try:
        if image_data and ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((size, size), Image.LANCZOS)
        arr = np.array(img) / 255.0
        return (arr > 0.5).astype(float)
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.zeros((size, size))


def matrix_to_transmission(matrix_data, size):
    try:
        matrix = np.array(matrix_data, dtype=float)
        if matrix.size == 0:
            return np.zeros((size, size))
        scale_y = size // matrix.shape[0]
        scale_x = size // matrix.shape[1]
        transmission = np.kron(matrix, np.ones((scale_y, scale_x)))
        result = np.zeros((size, size))
        my = min(size, transmission.shape[0])
        mx = min(size, transmission.shape[1])
        result[:my, :mx] = transmission[:my, :mx]
        return result
    except Exception as e:
        print(f"Error creating aperture from matrix: {e}")
        return np.zeros((size, size))


def apply_electron_statistics(intensity, n_electrons):
    prob = intensity.copy()
    prob_sum = np.sum(prob)
    if prob_sum > 0:
        prob /= prob_sum
    else:
        prob = np.ones_like(prob) / prob.size
    flat_prob = prob.flatten()
    indices = np.random.choice(len(flat_prob), size=n_electrons, p=flat_prob)
    result = np.zeros_like(flat_prob)
    np.add.at(result, indices, 1)
    result = result.reshape(intensity.shape)
    result = gaussian_filter(result, sigma=1.5)
    if np.max(result) > 0:
        result /= np.max(result)
    return result


def apply_gamma(arr, gamma=0.5):
    result = np.power(np.clip(arr, 0, None), gamma)
    if np.max(result) > 0:
        result /= np.max(result)
    return result


def array_to_png_base64(arr, colormap='hot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    normalized = arr / np.max(arr) if np.max(arr) > 0 else arr
    cmap = plt.cm.get_cmap(colormap)
    colored = cmap(normalized)
    uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(uint8, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def array_to_gray_png_base64(arr):
    normalized = arr / np.max(arr) if np.max(arr) > 0 else arr
    uint8 = (normalized * 255).astype(np.uint8)
    img = Image.fromarray(uint8, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        ap_type = data.get('aperture_type', 'single')
        distance = float(data.get('screen_distance', 1.0))
        n_elec = int(data.get('num_electrons', 100000))
        wavelength = float(data.get('wavelength', 5e-12))
        pixel_size = float(data.get('pixel_size', 1e-7))
        size = 512
        n_elec = max(10000, min(1000000, n_elec))
        distance = max(0.001, min(100.0, distance))
        
        aperture = create_aperture(
            ap_type,
            size_pixels=size,
            pixel_size=pixel_size,
            slit_width=int(data.get('slit_width', 20)),
            slit_separation=int(data.get('slit_separation', 60)),
            radius=int(data.get('circle_radius', 40)),
            n_slits=int(data.get('n_slits', 5)),
            image_data=data.get('image_data'),
            matrix_data=data.get('matrix_data')
        )
        
        I_fresnel = aperture.calculate_fresnel(wavelength, distance)
        I_fraunhofer = aperture.calculate_fraunhofer(wavelength)
        
        fresnel_elec = apply_electron_statistics(I_fresnel, n_elec)
        fraunhofer_elec = apply_electron_statistics(I_fraunhofer, n_elec)
        fresnel_disp = apply_gamma(fresnel_elec, 0.4)
        fraunhofer_disp = apply_gamma(fraunhofer_elec, 0.4)
        ap_img = array_to_gray_png_base64(aperture.get_transmission())
        fr_img = array_to_png_base64(fresnel_disp, 'hot')
        fh_img = array_to_png_base64(fraunhofer_disp, 'hot')
        char_size = aperture.physical_size / 4
        F = char_size**2 / (wavelength * distance)
        
        return jsonify({
            'success': True,
            'aperture': ap_img,
            'fresnel': fr_img,
            'fraunhofer': fh_img,
            'fresnel_number': float(F),
            'algorithm': aperture.__class__.__name__,
            'info': {
                'wavelength': wavelength,
                'distance': distance,
                'electrons': n_elec,
                'aperture_type': ap_type
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/calculate_analytical', methods=['POST'])
def calculate_analytical():
    try:
        data = request.get_json()
        ap_type = data.get('aperture_type', 'single')
        distance = float(data.get('screen_distance', 1.0))
        wavelength = float(data.get('wavelength', 5e-12))
        pixel_size = float(data.get('pixel_size', 1e-7))
        size = 512
        
        aperture = create_aperture(
            ap_type,
            size_pixels=size,
            pixel_size=pixel_size,
            slit_width=int(data.get('slit_width', 20)),
            slit_separation=int(data.get('slit_separation', 60)),
            radius=int(data.get('circle_radius', 40)),
            n_slits=int(data.get('n_slits', 5)),
            image_data=data.get('image_data'),
            matrix_data=data.get('matrix_data')
        )
        
        I_fresnel = aperture.calculate_fresnel(wavelength, distance)
        I_fraunhofer = aperture.calculate_fraunhofer(wavelength)
        
        fresnel_disp = apply_gamma(I_fresnel, 0.5)
        fraunhofer_disp = apply_gamma(I_fraunhofer, 0.5)
        ap_img = array_to_gray_png_base64(aperture.get_transmission())
        fr_img = array_to_png_base64(fresnel_disp, 'inferno')
        fh_img = array_to_png_base64(fraunhofer_disp, 'inferno')
        char_size = aperture.physical_size / 4
        F = char_size**2 / (wavelength * distance)
        
        return jsonify({
            'success': True,
            'aperture': ap_img,
            'fresnel': fr_img,
            'fraunhofer': fh_img,
            'fresnel_number': float(F),
            'algorithm': aperture.__class__.__name__ + ' (analytical)',
            'info': {
                'wavelength': wavelength,
                'distance': distance,
                'electrons': 0,
                'aperture_type': ap_type
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    import os
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)
