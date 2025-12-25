from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)

class DiffractionSimulator:
    def __init__(self):
        self.h = 6.626e-34
        self.m_e = 9.109e-31
        self.e = 1.602e-19
        self.default_energy_eV = 150

    def electron_wavelength(self, energy_eV):
        energy_J = energy_eV * self.e
        return self.h / np.sqrt(2 * self.m_e * energy_J)

    def create_single_slit(self, size, slit_width_pixels):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, slit_width_pixels // 2)
        aperture[:, center - half_width:center + half_width] = 1
        return aperture

    def create_double_slit(self, size, slit_width_pixels, separation_pixels):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, slit_width_pixels // 2)
        half_sep = separation_pixels // 2
        left_center = center - half_sep
        aperture[:, left_center - half_width:left_center + half_width] = 1
        right_center = center + half_sep
        aperture[:, right_center - half_width:right_center + half_width] = 1
        return aperture

    def create_triple_slit(self, size, slit_width_pixels, separation_pixels):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, slit_width_pixels // 2)
        aperture[:, center - half_width:center + half_width] = 1
        aperture[:, center - separation_pixels - half_width:center - separation_pixels + half_width] = 1
        aperture[:, center + separation_pixels - half_width:center + separation_pixels + half_width] = 1
        return aperture

    def create_circular_aperture(self, size, radius_pixels):
        aperture = np.zeros((size, size))
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius_pixels**2
        aperture[mask] = 1
        return aperture

    def create_from_image(self, image_data, size):
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('L')
        img = img.resize((size, size))
        aperture = np.array(img) / 255.0
        return (aperture > 0.5).astype(float)

    def create_from_grid(self, grid_data, size):
        grid = np.array(grid_data, dtype=float)
        grid_size = grid.shape[0]
        aperture = np.zeros((size, size))
        cell_size = size // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == 1:
                    aperture[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = 1
        return aperture

    def fresnel_diffraction(self, aperture, wavelength, z, pixel_size):
        N = aperture.shape[0]

        if z <= 0:
            return aperture.copy()

        k = 2 * np.pi / wavelength
        L = N * pixel_size

        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        X1, Y1 = np.meshgrid(x, y)

        L_out = wavelength * z / pixel_size
        x2 = np.linspace(-L_out/2, L_out/2, N)
        y2 = np.linspace(-L_out/2, L_out/2, N)
        X2, Y2 = np.meshgrid(x2, y2)

        c = np.exp(1j * k * z) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2)) / (1j * wavelength * z)

        U0 = aperture * np.exp(1j * k / (2 * z) * (X1**2 + Y1**2))

        U = c * fftshift(fft2(ifftshift(U0))) * pixel_size**2

        intensity = np.abs(U)**2

        return intensity

    def fraunhofer_diffraction(self, aperture):
        N = aperture.shape[0]

        pad_factor = 4
        pad_size = N * pad_factor
        padded = np.zeros((pad_size, pad_size))
        offset = (pad_size - N) // 2
        padded[offset:offset+N, offset:offset+N] = aperture

        diffracted = fftshift(fft2(ifftshift(padded)))
        intensity_full = np.abs(diffracted)**2

        center = pad_size // 2
        half_N = N // 2
        intensity = intensity_full[center-half_N:center+half_N, center-half_N:center+half_N]

        return intensity

    def add_quantum_noise(self, intensity, num_electrons):
        if num_electrons <= 0:
            return intensity

        prob = intensity.copy()
        prob_sum = np.sum(prob)
        if prob_sum == 0:
            return intensity

        prob = prob / prob_sum
        indices = np.random.choice(len(prob.flatten()), size=num_electrons, p=prob.flatten())

        result = np.zeros_like(intensity)
        np.add.at(result.ravel(), indices, 1)
        return result

    def normalize(self, arr):
        if np.max(arr) > 0:
            return arr / np.max(arr)
        return arr

    def apply_gamma(self, arr, gamma):
        return np.power(arr, gamma)

    def simulate(self, aperture_type, params, screen_distance_m,
                 num_electrons, grid_size, gamma_enabled, gamma,
                 custom_image=None, grid_data=None):

        wavelength = self.electron_wavelength(self.default_energy_eV)

        char_size_m = params.get('char_size_m', 10e-6)

        field_size_m = char_size_m * 20
        pixel_size_m = field_size_m / grid_size

        slit_width_px = max(2, int(params.get('slit_width_m', 10e-6) / pixel_size_m))
        separation_px = max(4, int(params.get('separation_m', 25e-6) / pixel_size_m))
        radius_px = max(2, int(params.get('radius_m', 20e-6) / pixel_size_m))

        if aperture_type == 'single':
            aperture = self.create_single_slit(grid_size, slit_width_px)
        elif aperture_type == 'double':
            aperture = self.create_double_slit(grid_size, slit_width_px, separation_px)
        elif aperture_type == 'triple':
            aperture = self.create_triple_slit(grid_size, slit_width_px, separation_px)
        elif aperture_type == 'circle':
            aperture = self.create_circular_aperture(grid_size, radius_px)
        elif aperture_type == 'image' and custom_image:
            aperture = self.create_from_image(custom_image, grid_size)
        elif aperture_type == 'grid' and grid_data:
            aperture = self.create_from_grid(grid_data, grid_size)
        else:
            aperture = self.create_single_slit(grid_size, slit_width_px)

        fresnel = self.fresnel_diffraction(aperture, wavelength, screen_distance_m, pixel_size_m)
        fraunhofer = self.fraunhofer_diffraction(aperture)

        fresnel = self.normalize(fresnel)
        fraunhofer = self.normalize(fraunhofer)

        if gamma_enabled:
            fresnel = self.apply_gamma(fresnel, gamma)
            fraunhofer = self.apply_gamma(fraunhofer, gamma)
            fresnel = self.normalize(fresnel)
            fraunhofer = self.normalize(fraunhofer)

        fresnel = self.add_quantum_noise(fresnel, num_electrons)
        fraunhofer = self.add_quantum_noise(fraunhofer, num_electrons)

        fresnel = self.normalize(fresnel)
        fraunhofer = self.normalize(fraunhofer)

        fresnel_number = (char_size_m ** 2) / (wavelength * screen_distance_m) if screen_distance_m > 0 else float('inf')

        return {
            'aperture': aperture,
            'fresnel': fresnel,
            'fraunhofer': fraunhofer,
            'wavelength_m': wavelength,
            'field_size_m': field_size_m,
            'pixel_size_m': pixel_size_m,
            'fresnel_number': fresnel_number
        }


simulator = DiffractionSimulator()


def array_to_base64(arr, title='', cmap='hot'):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, color='white', fontweight='bold')
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none', dpi=120, pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def convert_to_meters(value, unit):
    multipliers = {'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'cm': 1e-2, 'm': 1}
    return value * multipliers.get(unit, 1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json

    aperture_type = data.get('aperture_type', 'single')
    grid_size = int(data.get('grid_size', 256))
    num_electrons = int(data.get('num_electrons', 10000))
    gamma_enabled = data.get('gamma_enabled', False)
    gamma = float(data.get('gamma', 0.3))

    slit_width_m = convert_to_meters(
        float(data.get('slit_width', 10)),
        data.get('slit_width_unit', 'um')
    )
    separation_m = convert_to_meters(
        float(data.get('separation', 25)),
        data.get('separation_unit', 'um')
    )
    radius_m = convert_to_meters(
        float(data.get('radius', 20)),
        data.get('radius_unit', 'um')
    )
    screen_distance_m = convert_to_meters(
        float(data.get('screen_distance', 0.1)),
        data.get('distance_unit', 'm')
    )

    if aperture_type == 'single':
        char_size_m = slit_width_m
    elif aperture_type in ['double', 'triple']:
        char_size_m = separation_m
    elif aperture_type == 'circle':
        char_size_m = radius_m * 2
    else:
        char_size_m = 10e-6

    params = {
        'slit_width_m': slit_width_m,
        'separation_m': separation_m,
        'radius_m': radius_m,
        'char_size_m': char_size_m
    }

    custom_image = data.get('custom_image')
    grid_data = data.get('grid_data')

    result = simulator.simulate(
        aperture_type, params, screen_distance_m,
        num_electrons, grid_size, gamma_enabled, gamma,
        custom_image, grid_data
    )

    if screen_distance_m <= 0:
        fresnel_title = 'z = 0'
    else:
        fresnel_title = f'z = {screen_distance_m:.2g} m'

    aperture_img = array_to_base64(result['aperture'], 'Aperture', 'gray')
    fresnel_img = array_to_base64(result['fresnel'], fresnel_title, 'hot')
    fraunhofer_img = array_to_base64(result['fraunhofer'], 'Fraunhofer', 'hot')

    wavelength_nm = result['wavelength_m'] * 1e9
    field_size_um = result['field_size_m'] * 1e6

    fn = result['fresnel_number']
    fresnel_number_str = f'{fn:.2e}' if fn < 1000 and fn != float('inf') else ('∞' if fn == float('inf') else f'{fn:.0f}')

    return jsonify({
        'success': True,
        'aperture': aperture_img,
        'fresnel': fresnel_img,
        'fraunhofer': fraunhofer_img,
        'wavelength_nm': wavelength_nm,
        'field_size_um': field_size_um,
        'fresnel_number': fresnel_number_str,
        'energy_eV': simulator.default_energy_eV
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)