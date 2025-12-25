from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)

def wavelength_to_rgb(wavelength_nm):
    """Преобразует длину волны (в нм) в RGB цвет для видимого спектра"""
    # Для невидимого спектра используем условные цвета
    if wavelength_nm < 380:
        # Ультрафиолет -> фиолетовый
        return (0.5, 0.0, 1.0)
    elif wavelength_nm > 750:
        # Инфракрасный -> тёмно-красный
        return (0.5, 0.0, 0.0)

    # Видимый спектр
    if 380 <= wavelength_nm < 440:
        r = -(wavelength_nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength_nm < 490:
        r = 0.0
        g = (wavelength_nm - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength_nm < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm < 580:
        r = (wavelength_nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength_nm < 645:
        r = 1.0
        g = -(wavelength_nm - 645) / (645 - 580)
        b = 0.0
    else:  # 645-750
        r = 1.0
        g = 0.0
        b = 0.0

    # Уменьшение интенсивности на краях видимого спектра
    if 380 <= wavelength_nm < 420:
        factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 700 < wavelength_nm <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength_nm) / (750 - 700)
    else:
        factor = 1.0

    return (r * factor, g * factor, b * factor)


def create_wavelength_colormap(wavelength_nm):
    """Создаёт цветовую карту от чёрного до цвета длины волны"""
    rgb = wavelength_to_rgb(wavelength_nm)
    colors = [(0, 0, 0), rgb]  # от чёрного к цвету
    return LinearSegmentedColormap.from_list('wavelength', colors, N=256)


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

    def fresnel_diffraction(self, aperture, wavelength, z, pixel_size, num_electrons, gamma_enabled, gamma):
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

        if gamma_enabled:
            intensity = self.apply_gamma_correction(intensity, gamma)

        if num_electrons > 0:
            intensity = self.add_quantum_noise(intensity, num_electrons)

        return intensity

    def fraunhofer_diffraction(self, aperture, wavelength, pixel_size, num_electrons, gamma_enabled, gamma):
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

        if gamma_enabled:
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
                 gamma_enabled, gamma, custom_aperture=None, grid_data=None):

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
                                                pixel_size, num_electrons, gamma_enabled, gamma)
        far_pattern = self.fraunhofer_diffraction(aperture, wavelength,
                                                  pixel_size, num_electrons, gamma_enabled, gamma)

        return aperture, near_pattern, far_pattern, wavelength


simulator = DiffractionSimulator()


def array_to_base64(arr, field_size_um, wavelength_nm, cmap='hot', title='', use_wavelength_color=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [-field_size_um/2, field_size_um/2, -field_size_um/2, field_size_um/2]

    # Выбор цветовой карты
    if use_wavelength_color and cmap != 'gray':
        colormap = create_wavelength_colormap(wavelength_nm)
    else:
        colormap = cmap

    im = ax.imshow(arr, cmap=colormap, origin='lower', extent=extent)
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
    gamma_enabled = data.get('gamma_enabled', False)
    gamma = float(data.get('gamma', 0.3))
    use_wavelength_color = data.get('use_wavelength_color', True)

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
        gamma_enabled, gamma, custom_aperture, grid_data
    )

    field_size_um = aperture_size * pixel_size_um
    wavelength_nm = wavelength * 1e9

    if screen_distance <= 0:
        near_title = 'Плоскость апертуры (z = 0)'
    else:
        near_title = f'Дифракция Френеля (z = {screen_distance} м)'

    # Получаем цвет для отображения в информации
    rgb = wavelength_to_rgb(wavelength_nm)
    color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

    aperture_img = array_to_base64(aperture, field_size_um, wavelength_nm,
                                   cmap='gray', title='Апертура (щель)',
                                   use_wavelength_color=False)
    near_img = array_to_base64(near_pattern, field_size_um, wavelength_nm,
                               cmap='hot', title=near_title,
                               use_wavelength_color=use_wavelength_color)
    far_img = array_to_base64(far_pattern, field_size_um, wavelength_nm,
                              cmap='hot', title='Дифракция Фраунгофера (дальняя зона)',
                              use_wavelength_color=use_wavelength_color)

    return jsonify({
        'success': True,
        'aperture': aperture_img,
        'near_pattern': near_img,
        'far_pattern': far_img,
        'wavelength': wavelength * 1e12,
        'wavelength_nm': wavelength_nm,
        'field_size_um': field_size_um,
        'color_hex': color_hex
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)