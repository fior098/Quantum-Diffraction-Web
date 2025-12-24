from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import zoom
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Настройка для русского текста
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)


class DiffractionSimulator:
    def __init__(self):
        self.h = 6.626e-34      # Постоянная Планка
        self.m_e = 9.109e-31    # Масса электрона
        self.e = 1.602e-19      # Заряд электрона

    def electron_wavelength(self, energy_eV):
        """Вычисление длины волны де Бройля"""
        energy_J = energy_eV * self.e
        wavelength = self.h / np.sqrt(2 * self.m_e * energy_J)
        return wavelength

    def create_single_slit(self, size, slit_width):
        """Одиночная щель"""
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width * size / 2))
        aperture[:, center - half_width:center + half_width] = 1
        return aperture

    def create_double_slit(self, size, slit_width, separation):
        """Двойная щель"""
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width * size / 2))
        half_sep = int(separation * size / 2)
        aperture[:, center - half_sep - half_width:center - half_sep + half_width] = 1
        aperture[:, center + half_sep - half_width:center + half_sep + half_width] = 1
        return aperture

    def create_triple_slit(self, size, slit_width, separation):
        """Тройная щель"""
        aperture = np.zeros((size, size))
        center = size // 2
        half_width = max(1, int(slit_width * size / 2))
        sep = int(separation * size)
        aperture[:, center - half_width:center + half_width] = 1
        aperture[:, center - sep - half_width:center - sep + half_width] = 1
        aperture[:, center + sep - half_width:center + sep + half_width] = 1
        return aperture

    def create_circular_aperture(self, size, radius):
        """Круглая апертура"""
        aperture = np.zeros((size, size))
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= (radius * size)**2
        aperture[mask] = 1
        return aperture

    def create_from_image(self, image_data, size):
        """Апертура из изображения"""
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('L')
        img = img.resize((size, size))
        aperture = np.array(img) / 255.0
        aperture = (aperture > 0.5).astype(float)
        return aperture

    def create_from_matrix(self, matrix_str, size):
        """Апертура из матрицы"""
        rows = matrix_str.strip().split('\n')
        matrix = []
        for row in rows:
            values = [int(x) for x in row.strip().split()]
            matrix.append(values)
        matrix = np.array(matrix, dtype=float)
        if matrix.shape[0] != size or matrix.shape[1] != size:
            zoom_factor = size / max(matrix.shape)
            matrix = zoom(matrix, zoom_factor, order=0)
            matrix = matrix[:size, :size]
        return (matrix > 0.5).astype(float)

    def fresnel_diffraction(self, aperture, wavelength, z, pixel_size, num_electrons):
        """
        Дифракция Френеля (ближняя зона).
        Использует метод углового спектра для точного вычисления.
        При z=0 возвращает саму апертуру.
        """
        N = aperture.shape[0]

        # При z <= 0 возвращаем апертуру
        if z <= 0:
            intensity = aperture.copy()
            if np.max(intensity) > 0:
                intensity = intensity / np.max(intensity)
            if num_electrons > 0:
                intensity = self.add_quantum_noise(intensity, num_electrons)
            return intensity

        k = 2 * np.pi / wavelength

        # Пространственные частоты
        fx = np.fft.fftfreq(N, pixel_size)
        fy = np.fft.fftfreq(N, pixel_size)
        FX, FY = np.meshgrid(fx, fy)

        # Ограничение для предотвращения эванесцентных волн
        f_max = 1 / wavelength
        valid_mask = (FX**2 + FY**2) < f_max**2

        # Передаточная функция углового спектра
        H = np.zeros((N, N), dtype=complex)
        H[valid_mask] = np.exp(1j * k * z * np.sqrt(
            1 - (wavelength * FX[valid_mask])**2 - (wavelength * FY[valid_mask])**2
        ))

        # Применяем распространение
        field_fft = fft2(aperture)
        propagated_fft = field_fft * ifftshift(H)
        propagated_field = ifft2(propagated_fft)

        intensity = np.abs(propagated_field)**2

        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

        if num_electrons > 0:
            intensity = self.add_quantum_noise(intensity, num_electrons)

        return intensity

    def fraunhofer_diffraction(self, aperture, wavelength, pixel_size, num_electrons):
        """
        Дифракция Фраунгофера (дальняя зона).
        Интенсивность = |Фурье-образ апертуры|^2

        Для одиночной щели: I ~ sinc^2
        Для двойной щели: I ~ sinc^2 * cos^2
        """
        N = aperture.shape[0]

        # Добавляем нулевое дополнение для лучшего разрешения
        pad_factor = 4
        pad_size = N * pad_factor
        padded = np.zeros((pad_size, pad_size))
        offset = (pad_size - N) // 2
        padded[offset:offset+N, offset:offset+N] = aperture

        # Фурье-преобразование (сдвиги для центрирования)
        diffracted = fftshift(fft2(ifftshift(padded)))
        intensity = np.abs(diffracted)**2

        # Извлекаем центральную часть с увеличенным разрешением
        center = pad_size // 2
        half_N = N // 2
        intensity = intensity[center-half_N:center+half_N, center-half_N:center+half_N]

        # Нормализация
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

        if num_electrons > 0:
            intensity = self.add_quantum_noise(intensity, num_electrons)

        return intensity

    def add_quantum_noise(self, intensity, num_electrons):
        """Добавление квантового шума (статистика попаданий электронов)"""
        prob = intensity.copy()
        prob_sum = np.sum(prob)

        if prob_sum == 0:
            return intensity

        prob = prob / prob_sum
        flat_prob = prob.flatten()

        # Случайное распределение электронов по вероятности
        indices = np.random.choice(len(flat_prob), size=num_electrons, p=flat_prob)

        result = np.zeros_like(intensity)
        np.add.at(result.ravel(), indices, 1)

        if np.max(result) > 0:
            result = result / np.max(result)

        return result

    def simulate(self, aperture_type, params, screen_distance,
                 energy_eV, num_electrons, aperture_size, pixel_size_um,
                 custom_aperture=None, matrix_data=None):
        """Основная функция симуляции"""

        wavelength = self.electron_wavelength(energy_eV)
        pixel_size = pixel_size_um * 1e-6

        # Создание апертуры
        if aperture_type == 'single':
            aperture = self.create_single_slit(aperture_size, params.get('slit_width', 0.05))
        elif aperture_type == 'double':
            aperture = self.create_double_slit(aperture_size,
                                               params.get('slit_width', 0.02),
                                               params.get('separation', 0.1))
        elif aperture_type == 'triple':
            aperture = self.create_triple_slit(aperture_size,
                                               params.get('slit_width', 0.02),
                                               params.get('separation', 0.08))
        elif aperture_type == 'circle':
            aperture = self.create_circular_aperture(aperture_size, params.get('radius', 0.1))
        elif aperture_type == 'image' and custom_aperture:
            aperture = self.create_from_image(custom_aperture, aperture_size)
        elif aperture_type == 'matrix' and matrix_data:
            aperture = self.create_from_matrix(matrix_data, aperture_size)
        else:
            aperture = self.create_single_slit(aperture_size, 0.05)

        # Вычисление дифракционных картин
        near_pattern = self.fresnel_diffraction(aperture, wavelength, screen_distance,
                                                pixel_size, num_electrons)
        far_pattern = self.fraunhofer_diffraction(aperture, wavelength,
                                                  pixel_size, num_electrons)

        return aperture, near_pattern, far_pattern, wavelength


simulator = DiffractionSimulator()


def array_to_base64(arr, cmap='hot', title=''):
    """Конвертация массива в base64 изображение"""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap=cmap, origin='lower')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    ax.set_xlabel('Позиция X (пиксели)', color='white', fontsize=10)
    ax.set_ylabel('Позиция Y (пиксели)', color='white', fontsize=10)
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


def create_1d_profile(pattern, title=''):
    """Создание 1D профиля интенсивности"""
    fig, ax = plt.subplots(figsize=(8, 3))
    center = pattern.shape[0] // 2
    profile = pattern[center, :]
    x_pixels = np.arange(len(profile))

    ax.plot(x_pixels, profile, color='#00d4ff', linewidth=1.5)
    ax.fill_between(x_pixels, profile, alpha=0.3, color='#00d4ff')
    ax.set_title(title, fontsize=12, color='white', fontweight='bold')
    ax.set_xlabel('Позиция (пиксели)', color='white', fontsize=10)
    ax.set_ylabel('Интенсивность (отн. ед.)', color='white', fontsize=10)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')

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
    screen_distance = float(data.get('screen_distance', 1.0))
    energy_eV = float(data.get('energy', 100))
    num_electrons = int(data.get('num_electrons', 10000))
    aperture_size = int(data.get('aperture_size', 256))
    pixel_size_um = float(data.get('pixel_size', 1.0))

    params = {
        'slit_width': float(data.get('slit_width', 0.05)),
        'separation': float(data.get('separation', 0.1)),
        'radius': float(data.get('radius', 0.1))
    }

    custom_aperture = data.get('custom_image', None)
    matrix_data = data.get('matrix_data', None)

    aperture, near_pattern, far_pattern, wavelength = simulator.simulate(
        aperture_type, params, screen_distance,
        energy_eV, num_electrons, aperture_size, pixel_size_um,
        custom_aperture, matrix_data
    )

    # Формирование заголовков на русском
    if screen_distance <= 0:
        near_title = 'Плоскость апертуры (z = 0)'
    else:
        near_title = f'Дифракция Френеля (z = {screen_distance} м)'

    aperture_img = array_to_base64(aperture, cmap='gray', title='Апертура (щель)')
    near_img = array_to_base64(near_pattern, cmap='hot', title=near_title)
    far_img = array_to_base64(far_pattern, cmap='hot', title='Дифракция Фраунгофера (дальняя зона)')

    near_profile = create_1d_profile(near_pattern, 'Профиль интенсивности - ближняя зона')
    far_profile = create_1d_profile(far_pattern, 'Профиль интенсивности - дальняя зона')

    return jsonify({
        'success': True,
        'aperture': aperture_img,
        'near_pattern': near_img,
        'far_pattern': far_img,
        'near_profile': near_profile,
        'far_profile': far_profile,
        'wavelength': wavelength * 1e12,
        'wavelength_nm': wavelength * 1e9
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)