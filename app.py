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

    def electron_wavelength(self, energy_eV):
        energy_J = energy_eV * self.e
        wavelength = self.h / np.sqrt(2 * self.m_e * energy_J)
        return wavelength

    def create_single_slit(self, size, slit_width_um, pixel_size_um):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width_um / (2 * pixel_size_um)))
        aperture[:, center - half_width:center + half_width] = 1
        return aperture

    def create_double_slit(self, size, slit_width_um, separation_um, pixel_size_um):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width_um / (2 * pixel_size_um)))
        half_sep = int(separation_um / (2 * pixel_size_um))
        aperture[:, center - half_sep - half_width:center - half_sep + half_width] = 1
        aperture[:, center + half_sep - half_width:center + half_sep + half_width] = 1
        return aperture

    def create_triple_slit(self, size, slit_width_um, separation_um, pixel_size_um):
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width_um / (2 * pixel_size_um)))
        sep = int(separation_um / pixel_size_um)
        aperture[:, center - half_width:center + half_width] = 1
        aperture[:, center - sep - half_width:center - sep + half_width] = 1
        aperture[:, center + sep - half_width:center + sep + half_width] = 1
        return aperture

    def create_circular_aperture(self, size, radius_um, pixel_size_um):
        aperture = np.zeros((size, size))
        center = size // 2
        y, x = np.ogrid[:size, :size]
        r_pixels = radius_um / pixel_size_um
        mask = (x - center)**2 + (y - center)**2 <= r_pixels**2
        aperture[mask] = 1
        return aperture

    def create_from_image(self, image_data, size):
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('L')
        img = img.resize((size, size))
        aperture = np.array(img) / 255.0
        aperture = (aperture > 0.5).astype(float)
        return aperture

    def create_from_grid(self, grid_data, size):
        grid = np.array(grid_data, dtype=float)
        grid_size = grid.shape[0]
        aperture = np.zeros((size, size))
        cell_size = size // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == 1:
                    y_start = i * cell_size
                    y_end = (i + 1) * cell_size
                    x_start = j * cell_size
                    x_end = (j + 1) * cell_size
                    aperture[y_start:y_end, x_start:x_end] = 1
        return aperture

    def apply_gamma_correction(self, intensity, gamma):
        if gamma <= 0:
            gamma = 0.1
        corrected = np.power(intensity, gamma)
        if np.max(corrected) > 0:
            corrected = corrected / np.max(corrected)
        return corrected

    def fresnel_diffraction(self, aperture, wavelength, z, pixel_size, num_electrons, gamma):
        N = aperture.shape[0]

        if z <= 0:
            intensity = aperture.copy()
            if np.max(intensity) > 0:
                intensity = intensity / np.max(intensity)
            if num_electrons > 0:
                intensity = self.add_quantum_noise(intensity, num_electrons)
            return intensity

        k = 2 * np.pi / wavelength

        pad_factor = 4
        pad_size = N * pad_factor
        padded = np.zeros((pad_size, pad_size), dtype=complex)
        offset = (pad_size - N) // 2
        padded[offset:offset+N, offset:offset+N] = aperture

        fx = np.fft.fftfreq(pad_size, pixel_size)
        fy = np.fft.fftfreq(pad_size, pixel_size)
        FX, FY = np.meshgrid(fx, fy)

        f_max = 1 / wavelength
        valid_mask = (FX**2 + FY**2) < f_max**2

        H = np.zeros((pad_size, pad_size), dtype=complex)
        H[valid_mask] = np.exp(1j * k * z * np.sqrt(
            1 - (wavelength * FX[valid_mask])**2 - (wavelength * FY[valid_mask])**2
        ))

        field_fft = fft2(padded)
        propagated_fft = field_fft * ifftshift(H)
        propagated_field = ifft2(propagated_fft)

        intensity_full = np.abs(propagated_field)**2

        center = pad_size // 2
        half_N = N // 2
        intensity = intensity_full[center-half_N:center+half_N, center-half_N:center+half_N]

        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

        intensity = self.apply_gamma_correction(intensity, gamma)

        if num_electrons > 0:
            intensity = self.add_quantum_noise(intensity, num_electrons)

        return intensity

    def fraunhofer_diffraction(self, aperture, wavelength, pixel_size, num_electrons, gamma):
        N = aperture.shape[0]

        pad_factor = 8
        pad_size = N * pad_factor
        padded = np.zeros((pad_size, pad_size))
        offset = (pad_size - N) // 2
        padded[offset:offset+N, offset:offset+N] = aperture

        diffracted = fftshift(fft2(ifftshift(padded)))
        intensity_full = np.abs(diffracted)**2

        center = pad_size // 2
        half_N = N // 2
        intensity = intensity_full[center-half_N:center+half_N, center-half_N:center+half_N]

        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

        intensity = self.apply_gamma_correction(intensity, gamma)

        if num_electrons > 0:
            intensity = self.add_quantum_noise(intensity, num_electrons)

        return intensity

    def add_quantum_noise(self, intensity, num_electrons):
        prob = intensity.copy()
        prob_sum = np.sum(prob)

        if prob_sum == 0:
            return intensity

        prob = prob / prob_sum
        flat_prob = prob.flatten()

        indices = np.random.choice(len(flat_prob), size=num_electrons, p=flat_prob)

        result = np.zeros_like(intensity)
        np.add.at(result.ravel(), indices, 1)

        if np.max(result) > 0:
            result = result / np.max(result)

        return result

    def simulate(self, aperture_type, params, screen_distance,
                 energy_eV, num_electrons, aperture_size, pixel_size_um,
                 gamma, custom_aperture=None, grid_data=None):

        wavelength = self.electron_wavelength(energy_eV)
        pixel_size = pixel_size_um * 1e-6

        if aperture_type == 'single':
            aperture = self.create_single_slit(aperture_size, params.get('slit_width_um', 10), pixel_size_um)
        elif aperture_type == 'double':
            aperture = self.create_double_slit(aperture_size,
                                               params.get('slit_width_um', 10),
                                               params.get('separation_um', 25),
                                               pixel_size_um)
        elif aperture_type == 'triple':
            aperture = self.create_triple_slit(aperture_size,
                                               params.get('slit_width_um', 10),
                                               params.get('separation_um', 25),
                                               pixel_size_um)
        elif aperture_type == 'circle':
            aperture = self.create_circular_aperture(aperture_size, params.get('radius_um', 20), pixel_size_um)
        elif aperture_type == 'image' and custom_aperture:
            aperture = self.create_from_image(custom_aperture, aperture_size)
        elif aperture_type == 'grid' and grid_data:
            aperture = self.create_from_grid(grid_data, aperture_size)
        else:
            aperture = self.create_single_slit(aperture_size, 10, pixel_size_um)

        near_pattern = self.fresnel_diffraction(aperture, wavelength, screen_distance,
                                                pixel_size, num_electrons, gamma)
        far_pattern = self.fraunhofer_diffraction(aperture, wavelength,
                                                  pixel_size, num_electrons, gamma)

        return aperture, near_pattern, far_pattern, wavelength

simulator = DiffractionSimulator()

def array_to_base64(arr, field_size_um, cmap='hot', title=''):
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [-field_size_um/2, field_size_um/2, -field_size_um/2, field_size_um/2]
    im = ax.imshow(arr, cmap=cmap, origin='lower', extent=extent)
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    ax.set_xlabel('X (мкм)', color='white', fontsize=10)
    ax.set_ylabel('Y (мкм)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Интенсивность', color='white', fontsize=10)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json

    aperture_type = data.get('aperture_type', 'single')
    screen_distance = float(data.get('screen_distance', 0.1))
    energy_eV = float(data.get('energy', 100))
    num_electrons = int(data.get('num_electrons', 10000))
    aperture_size = int(data.get('aperture_size', 256))
    pixel_size_um = float(data.get('pixel_size', 1.0))
    gamma = float(data.get('gamma', 0.3))

    params = {
        'slit_width_um': float(data.get('slit_width_um', 10)),
        'separation_um': float(data.get('separation_um', 25)),
        'radius_um': float(data.get('radius_um', 20))
    }

    custom_aperture = data.get('custom_image', None)
    grid_data = data.get('grid_data', None)

    aperture, near_pattern, far_pattern, wavelength = simulator.simulate(
        aperture_type, params, screen_distance,
        energy_eV, num_electrons, aperture_size, pixel_size_um,
        gamma, custom_aperture, grid_data
    )

    field_size_um = aperture_size * pixel_size_um

    if screen_distance <= 0:
        near_title = 'Плоскость апертуры (z = 0)'
    else:
        near_title = f'Дифракция Френеля (z = {screen_distance} м)'

    aperture_img = array_to_base64(aperture, field_size_um, cmap='gray', title='Апертура (щель)')
    near_img = array_to_base64(near_pattern, field_size_um, cmap='hot', title=near_title)
    far_img = array_to_base64(far_pattern, field_size_um, cmap='hot', title='Дифракция Фраунгофера (дальняя зона)')

    return jsonify({
        'success': True,
        'aperture': aperture_img,
        'near_pattern': near_img,
        'far_pattern': far_img,
        'wavelength': wavelength * 1e12,
        'wavelength_nm': wavelength * 1e9,
        'field_size_um': field_size_um
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)