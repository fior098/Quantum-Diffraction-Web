from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from scipy.fft import fftn, ifftn, fft2, fftshift
from scipy.ndimage import zoom
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
import json
import imageio.v2 as imageio
import io

plt.rcParams['font.family'] = 'DejaVu Sans'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def create_2d_aperture_for_simulation(params, N, mask_path=None, matrix_str=None):
    preset = params.get('preset', 'single')
    slit_width = float(params.get('slit_width', 30))
    gap = float(params.get('gap', 80))

    scale = N / 256.0
    slit_width_px = max(2, int(slit_width * scale))
    gap_px = int(gap * scale)

    aperture = np.zeros((N, N), dtype=float)
    center = N // 2

    if preset == 'matrix' and matrix_str:
        matrix = json.loads(matrix_str)
        small = np.array(matrix, dtype=float)
        small_inverted = 1.0 - small
        aperture = zoom(small_inverted, N / small.shape[0], order=0)
        aperture = np.clip(aperture, 0, 1)
        return aperture

    if preset == 'image' and mask_path and os.path.exists(mask_path):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (N, N), interpolation=cv2.INTER_LANCZOS4)
            img = np.flipud(img)
            aperture = (img / 255.0).T
            return aperture

    if preset == 'single':
        half_w = slit_width_px // 2
        x_start = max(0, center - half_w)
        x_end = min(N, center + half_w)
        aperture[x_start:x_end, :] = 1.0

    elif preset == 'double':
        half_w = slit_width_px // 2
        half_g = gap_px // 2
        x1_start = max(0, center - half_g - half_w)
        x1_end = max(0, center - half_g + half_w)
        x2_start = min(N, center + half_g - half_w)
        x2_end = min(N, center + half_g + half_w)
        aperture[x1_start:x1_end, :] = 1.0
        aperture[x2_start:x2_end, :] = 1.0

    elif preset == 'triple':
        half_w = slit_width_px // 2
        aperture[center - half_w:center + half_w, :] = 1.0
        x1_start = max(0, center - gap_px - half_w)
        x1_end = max(0, center - gap_px + half_w)
        aperture[x1_start:x1_end, :] = 1.0
        x2_start = min(N, center + gap_px - half_w)
        x2_end = min(N, center + gap_px + half_w)
        aperture[x2_start:x2_end, :] = 1.0

    elif preset == 'circle':
        Y_grid, X_grid = np.ogrid[:N, :N]
        radius = slit_width_px / 2
        dist = np.sqrt((X_grid - center)**2 + (Y_grid - center)**2)
        aperture[dist <= radius] = 1.0

    return aperture


def create_2d_aperture_for_fraunhofer(params, N, mask_path=None, matrix_str=None):
    preset = params.get('preset', 'single')
    slit_width = float(params.get('slit_width', 30))
    gap = float(params.get('gap', 80))

    scale = N / 256.0
    slit_width_px = max(2, int(slit_width * scale))
    gap_px = int(gap * scale)

    aperture = np.zeros((N, N), dtype=float)
    center = N // 2

    if preset == 'matrix' and matrix_str:
        matrix = json.loads(matrix_str)
        small = np.array(matrix, dtype=float)
        small_inverted = 1.0 - small
        aperture = zoom(small_inverted, N / small.shape[0], order=0)
        aperture = np.clip(aperture, 0, 1)
        return aperture

    if preset == 'image' and mask_path and os.path.exists(mask_path):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (N, N), interpolation=cv2.INTER_LANCZOS4)
            aperture = img / 255.0
            return aperture

    if preset == 'single':
        half_w = slit_width_px // 2
        x_start = max(0, center - half_w)
        x_end = min(N, center + half_w)
        aperture[x_start:x_end, :] = 1.0

    elif preset == 'double':
        half_w = slit_width_px // 2
        half_g = gap_px // 2
        x1_start = max(0, center - half_g - half_w)
        x1_end = max(0, center - half_g + half_w)
        x2_start = min(N, center + half_g - half_w)
        x2_end = min(N, center + half_g + half_w)
        aperture[x1_start:x1_end, :] = 1.0
        aperture[x2_start:x2_end, :] = 1.0

    elif preset == 'triple':
        half_w = slit_width_px // 2
        aperture[center - half_w:center + half_w, :] = 1.0
        x1_start = max(0, center - gap_px - half_w)
        x1_end = max(0, center - gap_px + half_w)
        aperture[x1_start:x1_end, :] = 1.0
        x2_start = min(N, center + gap_px - half_w)
        x2_end = min(N, center + gap_px + half_w)
        aperture[x2_start:x2_end, :] = 1.0

    elif preset == 'circle':
        Y_grid, X_grid = np.ogrid[:N, :N]
        radius = slit_width_px / 2
        dist = np.sqrt((X_grid - center)**2 + (Y_grid - center)**2)
        aperture[dist <= radius] = 1.0

    return aperture


def get_preset_name_ru(preset):
    names = {
        'single': 'одна щель',
        'double': 'две щели',
        'triple': 'три щели',
        'circle': 'круглое отверстие',
        'matrix': 'матрица',
        'image': 'изображение'
    }
    return names.get(preset, preset)


def create_potential_3d(aperture_2d, N, z, z_mask, z_screen):
    V = np.zeros((N, N, N), dtype=float)
    V_max = 1e8

    barrier_thickness = max(2, N // 30)
    z_idx = np.argmin(np.abs(z - z_mask))
    z_start = max(0, z_idx - barrier_thickness // 2)
    z_end = min(N, z_idx + barrier_thickness // 2 + 1)

    for iz in range(z_start, z_end):
        V[:, :, iz] = np.where(aperture_2d > 0.5, 0.0, V_max)

    screen_idx = np.argmin(np.abs(z - z_screen))
    screen_thickness = max(2, N // 40)
    screen_start = max(0, screen_idx - screen_thickness // 2)
    screen_end = min(N, screen_idx + screen_thickness // 2 + 1)

    for iz in range(screen_start, screen_end):
        V[:, :, iz] = V_max

    return V, z_start, z_end, screen_start, screen_end


def create_absorbing_boundary_3d(N, width=10):
    absorb = np.ones((N, N, N), dtype=float)
    ramp = np.linspace(0, 1, width)**2

    for i in range(width):
        factor = ramp[i]
        absorb[i, :, :] *= factor
        absorb[N-1-i, :, :] *= factor
        absorb[:, i, :] *= factor
        absorb[:, N-1-i, :] *= factor
        absorb[:, :, i] *= factor
        absorb[:, :, N-1-i] *= factor

    return absorb


def run_dynamic_simulation_3d(params, mask_path=None, matrix_str=None):
    N = 80
    L = 100.0
    dx = dy = dz = L / N

    k_max = np.pi / dx

    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    k0_input = float(params.get('k0', 2.5))
    k0 = min(k0_input, 0.7 * k_max)

    sigma_x_input = float(params.get('sigma_x', 60))
    sigma_y_input = float(params.get('sigma_y', 60))
    sigma_z_input = float(params.get('sigma_z', 20))

    scale_factor = L / 256.0
    sigma_x = sigma_x_input * scale_factor
    sigma_y = sigma_y_input * scale_factor
    sigma_z = sigma_z_input * scale_factor

    z0_input = float(params.get('initial_z0', -80))
    z_mask_input = float(params.get('z_mask', 0))
    z_screen_input = float(params.get('z_screen', 80))

    z0 = z0_input * scale_factor
    z_mask = z_mask_input * scale_factor
    z_screen = z_screen_input * scale_factor

    z0 = max(-L/2 + sigma_z + 5, min(z0, z_mask - 10))
    z_screen = min(L/2 - 5, max(z_screen, z_mask + 10))

    aperture_2d = create_2d_aperture_for_simulation(params, N, mask_path, matrix_str)
    V, barrier_z_start, barrier_z_end, screen_z_start, screen_z_end = create_potential_3d(
        aperture_2d, N, z, z_mask, z_screen)

    gauss = np.exp(-X**2 / (2 * sigma_x**2)) * \
            np.exp(-Y**2 / (2 * sigma_y**2)) * \
            np.exp(-(Z - z0)**2 / (2 * sigma_z**2))

    phase = np.exp(1j * k0 * Z)
    psi = (gauss * phase).astype(np.complex128)

    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy * dz + 1e-20)
    psi /= norm

    v_group = k0
    distance = z_screen - z0
    t_arrival = distance / (v_group + 1e-10)
    t_total = t_arrival * 1.8

    dt = 0.08
    total_steps = int(t_total / dt) + 1
    total_steps = min(total_steps, 1500)
    save_every = max(1, total_steps // 45)

    absorb = create_absorbing_boundary_3d(N, width=8)

    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dy) * 2 * np.pi
    kz = np.fft.fftfreq(N, dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    kinetic = np.exp(-1j * dt * (KX**2 + KY**2 + KZ**2) / 2)

    V_clipped = np.clip(V, 0, 1e6)
    potential_half = np.exp(-1j * V_clipped * dt / 2)
    barrier_mask = V > 1e5

    screen_idx = np.argmin(np.abs(z - z_screen))
    detection_idx = max(0, screen_idx - 2)

    frames = []
    accumulated_screen = np.zeros((N, N), dtype=float)
    psi_current = psi.copy()

    barrier_xz = V[:, N//2, :] > 1e5

    preset = params.get('preset', 'single')
    preset_ru = get_preset_name_ru(preset)

    for step in range(total_steps):
        psi_current *= potential_half
        psi_current[barrier_mask] = 0.0

        psi_hat = fftn(psi_current)
        psi_hat *= kinetic
        psi_current = ifftn(psi_hat)

        psi_current *= potential_half
        psi_current[barrier_mask] = 0.0

        psi_current *= absorb

        screen_intensity = np.abs(psi_current[:, :, detection_idx])**2
        accumulated_screen += screen_intensity * dt

        current_time = step * dt

        if step % save_every == 0 or step == total_steps - 1:
            prob_xz = np.abs(psi_current[:, N//2, :])**2
            prob_max = prob_xz.max()
            prob_xz_norm = prob_xz / (prob_max + 1e-20)

            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(prob_xz_norm.T, cmap='hot', origin='lower',
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           vmin=0, vmax=1, aspect='auto')

            barrier_display = np.zeros((*barrier_xz.T.shape, 4))
            barrier_display[barrier_xz.T] = [0, 1, 1, 0.7]
            ax.imshow(barrier_display, origin='lower',
                      extent=[x.min(), x.max(), z.min(), z.max()],
                      aspect='auto')

            ax.axhline(y=z_screen, color='lime', linewidth=2, linestyle='-',
                       label=f'Экран (z={z_screen:.1f} нм)')
            ax.axhline(y=z_mask, color='cyan', linewidth=1, linestyle=':',
                       alpha=0.5, label=f'Щель (z={z_mask:.1f} нм)')

            ax.legend(loc='upper right', facecolor='black',
                      labelcolor='white', fontsize=9)

            progress = (step / total_steps) * 100
            ax.set_title(f'Время: {current_time:.1f} фс | {preset_ru} | Прогресс: {progress:.0f}%',
                         color='white', fontsize=12)
            ax.set_xlabel('X (нм)', color='white', fontsize=11)
            ax.set_ylabel('Z — направление распространения (нм)', color='white', fontsize=11)
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Плотность вероятности |ψ|²', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='black')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.imread(buf))

    gif_name = f"anim_{uuid.uuid4().hex}.gif"
    gif_path = os.path.join(app.config['RESULTS_FOLDER'], gif_name)
    imageio.mimsave(gif_path, frames, duration=0.15, loop=0)

    return (gif_name, os.path.join('results', gif_name),
            x, y, accumulated_screen, z_screen, aperture_2d)


def create_numerical_diffraction_image_2d(x, y, intensity_2d, z_screen, has_aperture):
    if intensity_2d is None or intensity_2d.size == 0:
        return None

    max_intensity = intensity_2d.max()
    if not has_aperture or max_intensity < 1e-20:
        return None

    intensity_norm = intensity_2d / max_intensity

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1],
                          hspace=0.25, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(intensity_norm.T, cmap='hot', origin='lower',
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=0, vmax=1, aspect='equal')
    ax1.set_xlabel('X (нм)', color='white', fontsize=12)
    ax1.set_ylabel('Y (нм)', color='white', fontsize=12)
    ax1.set_title(f'2D дифракционная картина на экране (Z = {z_screen:.1f} нм)\n'
                  f'Численное решение 3D уравнения Шрёдингера',
                  color='white', fontsize=14)
    ax1.tick_params(colors='white')
    ax1.set_facecolor('black')

    cbar = plt.colorbar(im, ax=ax1, shrink=0.7, pad=0.02)
    cbar.set_label('Интенсивность (накопленная)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax2 = fig.add_subplot(gs[1, 0])
    profile_x = intensity_norm.sum(axis=1)
    profile_x /= profile_x.max() + 1e-20
    ax2.plot(x, profile_x, 'lime', linewidth=2)
    ax2.fill_between(x, 0, profile_x, alpha=0.3, color='lime')
    ax2.set_xlabel('X (нм)', color='white')
    ax2.set_ylabel('I (сумма по Y)', color='white')
    ax2.set_title('Профиль вдоль оси X', color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('black')
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, color='gray')

    ax3 = fig.add_subplot(gs[1, 1])
    profile_y = intensity_norm.sum(axis=0)
    profile_y /= profile_y.max() + 1e-20
    ax3.plot(y, profile_y, 'cyan', linewidth=2)
    ax3.fill_between(y, 0, profile_y, alpha=0.3, color='cyan')
    ax3.set_xlabel('Y (нм)', color='white')
    ax3.set_ylabel('I (сумма по X)', color='white')
    ax3.set_title('Профиль вдоль оси Y', color='white')
    ax3.tick_params(colors='white')
    ax3.set_facecolor('black')
    ax3.set_xlim(y.min(), y.max())
    ax3.set_ylim(0, 1.15)
    ax3.grid(True, alpha=0.3, color='gray')

    fig.patch.set_facecolor('black')

    num_name = f"numerical_2d_{uuid.uuid4().hex}.png"
    num_path = os.path.join(app.config['RESULTS_FOLDER'], num_name)
    fig.savefig(num_path, bbox_inches='tight', dpi=150, facecolor='black')
    plt.close(fig)

    return os.path.join('results', num_name)


def compute_fraunhofer_diffraction(aperture):
    N = aperture.shape[0]
    has_aperture = np.sum(aperture) > 0

    if not has_aperture:
        return None, None

    padded = np.zeros((N * 2, N * 2), dtype=float)
    padded[N//2:N//2+N, N//2:N//2+N] = aperture

    ft = fftshift(fft2(fftshift(padded)))
    intensity = np.abs(ft)**2
    intensity /= intensity.max() + 1e-20

    return intensity, aperture


def check_has_aperture(preset, matrix_str=None, mask_path=None):
    if preset == 'matrix' and matrix_str:
        matrix = json.loads(matrix_str)
        small = np.array(matrix, dtype=float)
        return np.any(small < 0.5)
    if preset == 'image' and mask_path:
        return True
    if preset in ['single', 'double', 'triple', 'circle']:
        return True
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    preset = request.form.get('preset', 'single')

    params = {
        'preset': preset,
        'slit_width': request.form.get('slit_width', '30'),
        'gap': request.form.get('gap', '80'),
        'z_mask': request.form.get('z_mask', '0'),
        'k0': request.form.get('k0', '2.5'),
        'sigma_x': request.form.get('sigma_x', '60'),
        'sigma_y': request.form.get('sigma_y', '60'),
        'sigma_z': request.form.get('sigma_z', '20'),
        'initial_z0': request.form.get('initial_z0', '-80'),
        'z_screen': request.form.get('z_screen', '80'),
    }

    matrix_str = None
    if preset == 'matrix':
        matrix_str = request.form.get('matrix')

    mask_path = None
    if preset == 'image' and 'mask_image' in request.files:
        file = request.files['mask_image']
        if file and file.filename:
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(mask_path)

    has_aperture = check_has_aperture(preset, matrix_str, mask_path)

    gif_name, gif_rel, x_coords, y_coords, screen_intensity_2d, z_screen, aperture_2d = \
        run_dynamic_simulation_3d(params, mask_path, matrix_str)

    num_rel = None
    if screen_intensity_2d is not None:
        num_rel = create_numerical_diffraction_image_2d(
            x_coords, y_coords, screen_intensity_2d, z_screen, has_aperture)

    aperture_fraunhofer = create_2d_aperture_for_fraunhofer(params, 512, mask_path, matrix_str)
    intensity, aperture_vis = compute_fraunhofer_diffraction(aperture_fraunhofer)

    diff_rel = None
    if intensity is not None:
        N_int = intensity.shape[0]
        intensity_log = np.log1p(intensity * 1e6)
        intensity_log /= intensity_log.max() + 1e-20

        crop_size = N_int // 4
        center = N_int // 2
        intensity_crop = intensity_log[center-crop_size:center+crop_size,
        center-crop_size:center+crop_size]

        N_ap = aperture_vis.shape[0]
        crop_ap = N_ap // 8
        aperture_crop = aperture_vis[crop_ap:-crop_ap, crop_ap:-crop_ap]

        preset_ru = get_preset_name_ru(preset)

        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[10, 2], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(intensity_crop, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Дифракция Фраунгофера ({preset_ru})\nАналитическое решение (дальняя зона)',
                      color='white', fontsize=14)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(aperture_crop, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Апертура (белое = открыто)', color='white', fontsize=12)
        ax2.axis('off')

        fig.patch.set_facecolor('black')

        diff_name = f"diff_{uuid.uuid4().hex}.png"
        diff_path = os.path.join(app.config['RESULTS_FOLDER'], diff_name)
        fig.savefig(diff_path, bbox_inches='tight', dpi=200, facecolor='black')
        plt.close(fig)
        diff_rel = os.path.join('results', diff_name)

    result = {'gif_url': '/static/' + gif_rel}
    if diff_rel:
        result['diff_url'] = '/static/' + diff_rel
    if num_rel:
        result['num_url'] = '/static/' + num_rel

    return jsonify(result)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(port=5000, debug=True)