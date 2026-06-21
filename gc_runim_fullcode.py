"""#### GC-RUNIM Optimizer"""

import numpy as np
import pandas as pd
from mealpy import FloatVar, Problem, BES, HGS, RUN, GA, WOA, FOX
import time
import warnings

warnings.filterwarnings('ignore')

class FinalHybridOptimizer:

    def __init__(self, epoch=100, pop_size=30, switch_ratio=0.5,
                 early_stop_tol=1e-6, stagnation_limit=8):

        self.epoch = epoch
        self.pop_size = pop_size
        self.switch_ratio = switch_ratio

        self.GA_epochs = int(epoch * switch_ratio)
        self.run_epochs = epoch - self.GA_epochs

        self.early_stop_tol = early_stop_tol
        self.stagnation_limit = stagnation_limit
        self.history = type('History', (), {'list_global_best_fit': []})()

    @staticmethod
    def _get_best(model):
        if hasattr(model, "g_best") and model.g_best is not None:
            return model.g_best
        if hasattr(model, "solution_best") and model.solution_best is not None:
            return model.solution_best
        # Fallback: Popülasyonu fitness'a göre sırala
        model.pop.sort(key=lambda a: a.target.fitness)
        return model.pop[0]

    def solve(self, problem_dict):
        print(f"GA Phase: {self.GA_epochs} | RUN Phase: {self.run_epochs}")
        ga_model = GA.OriginalGA(epoch=1, pop_size=self.pop_size)
        ga_model.solve(problem_dict, termination={"max_epoch": 1})

        best = self._get_best(ga_model)
        prev_best = best.target.fitness
        stagnation = 0

        for t in range(1, self.GA_epochs + 1):
            ga_model.evolve(t)
            best = self._get_best(ga_model)
            self.history.list_global_best_fit.append(best.target.fitness)

            # Controlled Cauchy Jump
            if t % 3 == 0:
                jump_strength = 0.9 * (1 - t / self.GA_epochs)
                step = np.random.standard_cauchy(ga_model.problem.n_dims)
                x_new = ga_model.problem.correct_solution(best.solution + step * jump_strength)
                target_new = ga_model.problem.get_target(x_new)

                if target_new.fitness < best.target.fitness:
                    best.solution = x_new
                    best.target = target_new

            # Early stopping
            improvement = abs(prev_best - best.target.fitness)
            if improvement < self.early_stop_tol:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation >= self.stagnation_limit:
                print(f"GA early stop @ epoch {t}")
                break
            prev_best = best.target.fitness

        ga_result = best
        if self.run_epochs <= 0:
            return self._wrap_result(ga_result)

        # Elite transfer
        ga_model.pop.sort(key=lambda a: a.target.fitness)
        elite_size = max(1, int(self.pop_size * 0.3))
        elite_pop = ga_model.pop[:elite_size]

        class EnhancedRUN(RUN.OriginalRUN):
            def __init__(self, elite, parent_epoch, total_epoch, **kwargs):
                super().__init__(**kwargs)
                self.elite = elite
                self.parent_epoch = parent_epoch
                self.total_epoch = total_epoch

            def initialization(self):
                super().initialization()
                # Elitleri popülasyona enjekte et
                for i in range(min(len(self.elite), len(self.pop))):
                    self.pop[i].solution = self.elite[i].solution.copy()
                    self.pop[i].target = self.elite[i].target

            def evolve(self, epoch):
                super().evolve(epoch)
                global_epoch = self.parent_epoch + epoch
                best_in_run = FinalHybridOptimizer._get_best(self)

                # Adaptive Intensive Mutation
                if global_epoch > self.total_epoch * 0.5:
                    progress = (global_epoch - 0.5 * self.total_epoch) / (0.5 * self.total_epoch)
                    mutation_rate = 0.02 + 0.03 * progress
                    n_mutate = max(1, int(len(self.pop) * (0.3 - 0.2 * progress)))

                    for i in range(n_mutate):
                        agent = self.pop[i]
                        temp = agent.solution.copy()
                        n_changes = max(1, int(self.problem.n_dims * mutation_rate))
                        idx = np.random.choice(self.problem.n_dims, n_changes, replace=False)
                        temp[idx] += np.random.normal(0, 0.1, n_changes)
                        x_new = self.problem.correct_solution(temp)
                        target_new = self.problem.get_target(x_new)
                        if target_new.fitness < agent.target.fitness:
                            agent.solution = x_new
                            agent.target = target_new


                if global_epoch % 2 == 0 and global_epoch > self.total_epoch * 0.5:
                    idx = np.random.randint(len(self.pop))
                    f = 0.008 * np.exp(-3 * global_epoch / self.total_epoch)
                    r = np.random.uniform(-1, 1, self.problem.n_dims)
                    x_new = self.problem.correct_solution(
                        best_in_run.solution + r * f * (best_in_run.solution - self.pop[idx].solution)
                    )
                    target_new = self.problem.get_target(x_new)
                    if target_new.fitness < best_in_run.target.fitness:
                        best_in_run.solution = x_new
                        best_in_run.target = target_new

        run_model = EnhancedRUN(
            elite=elite_pop,
            parent_epoch=self.GA_epochs,
            total_epoch=self.epoch,
            epoch=self.run_epochs,
            pop_size=int(self.pop_size * 0.7)
        )

        run_res = run_model.solve(problem_dict)

        self.history.list_global_best_fit.extend(run_model.history.list_global_best_fit)

        if run_res.target.fitness < ga_result.target.fitness:
            print(f"RUN improved solution: {run_res.target.fitness:.6f}")
            return self._wrap_result(run_res)

        print(f"GA solution retained: {ga_result.target.fitness:.6f}")
        return self._wrap_result(ga_result)

    @staticmethod
    def _wrap_result(agent):
        class Result:
            def __init__(self, sol, fit):
                self.solution = sol
                self.target = type("Target", (), {"fitness": fit})()
        return Result(agent.solution, agent.target.fitness)

"""#### GC-RUNIM"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp2d
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
from mealpy import FloatVar, Problem, BES, HGS, RUN, GA, WOA, FOX
import time
import warnings
warnings.filterwarnings('ignore')
import time


import numpy as np
from pathlib import Path
from scipy.interpolate import interp2d
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class Sentinel2DataLoader:
    """
    Folder:
    Dataset/
        ├── Image1/
        │   ├── B2.tif
        │   ├── B3.tif
        │   └── ...
        ├── Image2/
        └── Image3/
    """

    BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B11', 'B12']
    BAND_RESOLUTIONS = {
        'B2': 10, 'B3': 10, 'B4': 10, 'B5': 20,
        'B8': 10, 'B8A': 20, 'B11': 20, 'B12': 20
    }
    TARGET_RESOLUTION = 10
    def __init__(self, dataset_folder: str):
        self.dataset_folder = Path(dataset_folder)
        self.bands = {}
        self.normalized_bands = {}
        self.available_images = self._scan_dataset()

    def _scan_dataset(self) -> List[str]:

        if not self.dataset_folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_folder}")


        image_folders = [f.name for f in self.dataset_folder.iterdir()
                        if f.is_dir() and f.name.startswith('Image')]


        if not image_folders:
            all_folders = [f.name for f in self.dataset_folder.iterdir() if f.is_dir()]

            print(f"\nFounded folders ({len(all_folders)}):")
            for folder in all_folders:
                folder_path = self.dataset_folder / folder
                tif_files = list(folder_path.glob("*.tif*"))
                print(f"   - {folder} ({len(tif_files)} .tif dosyası)")


            image_folders = []
            for folder in all_folders:
                folder_path = self.dataset_folder / folder

                has_bands = any(
                    list(folder_path.glob(f"{band}.*")) or
                    list(folder_path.glob(f"{band.replace('B', 'B0')}.*")) or
                    list(folder_path.glob(f"*{band}*.tif*"))
                    for band in ['B2', 'B3', 'B4']
                )
                if has_bands:
                    image_folders.append(folder)

            if image_folders:
                print(f"\n {len(image_folders)}")
                for img in image_folders:
                    print(f"   - {img}")
            else:
                print("\n Bands not found")


        else:
            image_folders.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            print(f"{len(image_folders)} images founded")
            for img in image_folders:
                print(f"   - {img}")

        return image_folders

    def load_scene(self, image_folder: str) -> Dict[str, np.ndarray]:
        """
        Load images

        Args:
            image_folder:
        """
        scene_path = self.dataset_folder / image_folder

        if not scene_path.exists():
            raise FileNotFoundError(f"Image file not found {scene_path}")

        self.bands = {}

        for band_name in self.BAND_NAMES:
            band_files = list(scene_path.glob(f"{band_name}.*")) + \
                        list(scene_path.glob(f"{band_name.replace('B', 'B0')}.*")) + \
                        list(scene_path.glob(f"*{band_name}*.tif*"))

            if not band_files:
                raise FileNotFoundError(
                    f" {band_name} band not found: {scene_path}\n"
                    f"   Required format: {band_name}.tif, {band_name}.tiff"
                )

            band_path = band_files[0]
            band_data = imread(str(band_path))
            self.bands[band_name] = band_data
            print(f"   ✓ {band_name}: {band_data.shape} - {band_path.name}")

        print(f" {len(self.bands)} band loaded succesfully: {image_folder}")
        return self.bands

    def normalize_bands(self) -> Dict[str, np.ndarray]:

        for band_name, band_data in self.bands.items():
            min_val = band_data.min()
            max_val = band_data.max()

            if max_val > min_val:
                normalized = (band_data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(band_data, dtype=np.float32)

            self.normalized_bands[band_name] = normalized.astype(np.float32)
        return self.normalized_bands

    def align_resolutions(self) -> Dict[str, np.ndarray]:
        ref_band = self.normalized_bands['B2']
        target_shape = ref_band.shape

        aligned_bands = {}

        for band_name, band_data in self.normalized_bands.items():
            if self.BAND_RESOLUTIONS[band_name] == 20:
                aligned = resize(band_data, target_shape,
                               order=1, mode='reflect',
                               anti_aliasing=True, preserve_range=True)
                aligned_bands[band_name] = aligned.astype(np.float32)
                print(f"🔧 {band_name}: {band_data.shape} → {aligned.shape}")
            else:
                aligned_bands[band_name] = band_data

        self.normalized_bands = aligned_bands
        return aligned_bands

    def get_band_stack(self) -> np.ndarray:
        """Returns all bands as a 3D array (H, W, C)"""
        band_list = [self.normalized_bands[name] for name in self.BAND_NAMES]
        return np.stack(band_list, axis=-1)


class DynamicBandFusion:
    """
    It finds the optimal band weights using a metaheuristic algorithm.
    Objective: Maximum Global Entropy or Variance
    """

    def __init__(self, band_stack: np.ndarray, metric='entropy'):
        """
        Args:
            band_stack: (H, W, C)
            metric: 'entropy' veya 'variance'
        """
        self.band_stack = band_stack
        self.n_bands = band_stack.shape[-1]
        self.metric = metric
        self.best_weights = None
        self.best_fusion = None

    def compute_entropy(self, image: np.ndarray, bins=256) -> float:
        """Global entropy """
        hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, 1), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def compute_variance(self, image: np.ndarray) -> float:
        """Global variance"""
        return np.var(image)

    def fitness_function(self, weights: np.ndarray) -> float:
        weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
        hybrid_image = np.tensordot(self.band_stack, weights, axes=([-1], [0]))
        hybrid_image = np.clip(hybrid_image, 0, 1)

        if self.metric == 'entropy':
            score = self.compute_entropy(hybrid_image)
        else:
            score = self.compute_variance(hybrid_image)
        return -score

    def create_fusion_image(self, weights: np.ndarray) -> np.ndarray:
        weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-10)
        fusion = np.tensordot(self.band_stack, weights, axes=([-1], [0]))
        return np.clip(fusion, 0, 1)


class OtsuMultiThreshold:
    def __init__(self, n_thresholds=5, n_bins=256):
        self.n_thresholds = n_thresholds
        self.n_bins = n_bins

    def compute_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hist, bin_edges = np.histogram(image.ravel(), bins=self.n_bins,
                                       range=(0, 1), density=False)
        hist = hist.astype(np.float64) / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return hist, bin_centers

    def between_class_variance(self, hist: np.ndarray, bin_centers: np.ndarray,
                               thresholds: np.ndarray) -> float:
        """
                Calculate between-class variance (σ_B^2)

        Args:
            hist: Normalized histogram
            bin_centers: Histogram bin centers
            thresholds: k threshold value (in the range 0–1)

        Returns:
            Between-class variance
        """
        thresholds_sorted = np.sort(np.clip(thresholds, 0.01, 0.99))
        threshold_bins = np.searchsorted(bin_centers, thresholds_sorted)
        threshold_bins = np.clip(threshold_bins, 0, self.n_bins - 1)
        boundaries = [0] + threshold_bins.tolist() + [self.n_bins]
        mu_global = np.sum(bin_centers * hist)
        sigma_b_squared = 0.0

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            w_k = np.sum(hist[start:end])

            if w_k > 1e-10:
                mu_k = np.sum(bin_centers[start:end] * hist[start:end]) / w_k
                sigma_b_squared += w_k * (mu_k - mu_global) ** 2

        return sigma_b_squared

    def fitness_function(self, thresholds: np.ndarray, hist: np.ndarray,
                        bin_centers: np.ndarray) -> float:

        thresholds = np.clip(thresholds, 0, 1)
        variance = self.between_class_variance(hist, bin_centers, thresholds)
        return -variance

    def apply_thresholds(self, image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        thresholds_sorted = np.sort(thresholds)
        segmented = np.zeros_like(image, dtype=np.uint8)

        for i, t in enumerate(thresholds_sorted):
            segmented[image > t] = i + 1

        return segmented


class SegmentationEvaluator:
    @staticmethod
    def compute_unsupervised_metrics(segmented: np.ndarray, original_fusion: np.ndarray) -> Dict[str, float]:
        """
        Liu's F-measure and Region-based Contrast.
        """
        N = original_fusion.size
        unique_labels = np.unique(segmented)
        weighted_var_sum = 0
        for label in unique_labels:
            mask = (segmented == label)
            n_i = np.sum(mask)
            if n_i > 1:
                weighted_var_sum += (np.sqrt(n_i) * np.var(original_fusion[mask]))

        f_measure = (1.0 / N) * weighted_var_sum * 100000

        # Inter-Class Contrast
        means = [original_fusion[segmented == l].mean() for l in unique_labels]
        contrast = np.std(means) if len(means) > 1 else 0

        return {"F-measure": f_measure, "Inter-Class Contrast": contrast}

    @staticmethod
    def compute_spectral_fidelity(original_stack: np.ndarray, fusion: np.ndarray) -> Dict[str, float]:
        ref = np.mean(original_stack, axis=-1).astype(np.float32)
        fusion = fusion.astype(np.float32)

        psnr_val = peak_signal_noise_ratio(ref, fusion, data_range=1.0)
        ssim_val = structural_similarity(ref, fusion, data_range=1.0)

        return {"Spectral-PSNR": psnr_val, "Spectral-SSIM": ssim_val}


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HistogramAnalyzer:
    """
    Advanced statistical visualization for multi-level segmentation.
    """

    @staticmethod
    def plot_segmentation_histogram(fusion_image: np.ndarray,
                                    segmented: np.ndarray,
                                    thresholds: np.ndarray,
                                    save_path: str = None):

        sns.set_context("paper", font_scale=1.2)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix' # LaTeX style math fonts

        fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
        unique_labels = np.unique(segmented)
        sorted_thresholds = np.sort(thresholds)


        main_color = "#34495e" # Dark Blue-Grey
        threshold_color = "#e74c3c" # Academic Red
        # Using Seaborn's colorblind friendly palettes
        class_palette = sns.color_palette("viridis", len(unique_labels))

        # --- (A) GLOBAL HISTOGRAM & OPTIMAL THRESHOLDS ---
        ax1 = axes[0, 0]
        ax1.hist(fusion_image.ravel(), bins=256, color=main_color, alpha=0.3,
                 density=True, label='Pixel Density')

        # Smooth kernel density estimate for a more "scientific" look
        sns.kdeplot(fusion_image.ravel(), ax=ax1, color=main_color, linewidth=2)

        for i, t in enumerate(sorted_thresholds):
            ax1.axvline(t, color=threshold_color, linestyle='--', linewidth=1.5,
                        label=f'$T_{{{i+1}}}: {t:.3f}$')

        ax1.set_title(r"$\mathbf{(a)}$ $\text{Global Intensity Distribution with Optimal Thresholds}$", pad=15)
        ax1.set_xlabel("Normalized Intensity $[0, 1]$",fontdict={'size': 16})
        ax1.set_ylabel("Probability Density",fontdict={'size': 16})
        ax1.legend(frameon=True,facecolor='white',framealpha=0.9,loc='upper left',bbox_to_anchor=(0.97, 1),fontsize=12)


        # --- (B) CLASS-WISE SPECTRAL OVERLAP ---
        ax2 = axes[0, 1]
        for i, label in enumerate(unique_labels):
            data = fusion_image[segmented == label]
            sns.kdeplot(data, ax=ax2, fill=True, color=class_palette[i],
                        label=f'Class {label}', alpha=0.4, linewidth=1.5)

        ax2.set_title(r"$\mathbf{(b)}$ $\text{Class-specific Spectral Signature & Overlap}$", pad=15)
        ax2.set_xlabel("Normalized Intensity",fontdict={'size': 16})
        ax2.set_ylabel("Density",fontdict={'size': 16})
        ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=11, frameon=True)

        # --- (C) QUANTITATIVE STATISTICAL CHARACTERIZATION ---
        ax3 = axes[1, 0]
        means = [np.mean(fusion_image[segmented == l]) for l in unique_labels]
        stds = [np.std(fusion_image[segmented == l]) for l in unique_labels]
        x_pos = [f"$C_{{{l}}}$" for l in unique_labels]

        # Use barplot with error bars
        bars = ax3.bar(x_pos, means, yerr=stds, color=class_palette, alpha=0.8,
                       edgecolor='black', linewidth=1, capsize=8)

        ax3.set_title(r"$\mathbf{(c)}$ $\text{Statistical Class Profiling (Mean } \pm \text{ Std)}$", pad=15)
        ax3.set_ylabel("Average Reflectance",fontdict={'size': 16})
        ax3.tick_params(axis='both', which='major', labelsize=13)
        ax3.set_ylim(0, 1.1)

        # --- (D) MULTI-CLASS CUMULATIVE DISTRIBUTION (CDF) ---
        ax4 = axes[1, 1]
        # Global CDF
        sorted_data = np.sort(fusion_image.ravel())
        y_vals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        ax4.plot(sorted_data, y_vals, color='black', linewidth=2.5, label='Global CDF', zorder=5)

        for i, label in enumerate(unique_labels):
            c_data = np.sort(fusion_image[segmented == label].ravel())
            if len(c_data) > 0:
                c_y = np.arange(len(c_data)) / float(len(c_data) - 1)
                ax4.plot(c_data, c_y, color=class_palette[i], linestyle='-.',
                         linewidth=1.5, label=f'Class {label} CDF')

        ax4.set_title(r"$\mathbf{(d)}$ $\text{Cumulative Information Capture Analysis}$", pad=15)
        ax4.set_xlabel("Intensity Threshold",fontdict={'size': 16})
        ax4.set_ylabel("Cumulative Probability",fontdict={'size': 16})
        ax4.legend(loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=11, frameon=True)

        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.subplots_adjust(right=0.82, wspace=0.4, hspace=0.3)

        if save_path:
            plt.savefig(save_path, dpi=400, format='pdf', bbox_inches='tight') # Standard for Q1
            print(f"figure saved to: {save_path}")

        plt.show()

class ConvergenceAnalyzer:
    def __init__(self):
        self.history = []

    def record_run(self, fitness_history: List[float]):
        self.history.append(fitness_history)

    def compute_statistics(self) -> Dict[str, float]:
        if not self.history:
            return {}

        best_fitness_per_run = [min(run) for run in self.history]

        stats = {
            'mean_best_fitness': np.mean(best_fitness_per_run),
            'std_best_fitness': np.std(best_fitness_per_run),
            'min_fitness': np.min(best_fitness_per_run),
            'max_fitness': np.max(best_fitness_per_run),
            'median_fitness': np.median(best_fitness_per_run)
        }

        return stats

    def plot_convergence(self, save_path=None):
        if not self.history:
            return

        plt.figure(figsize=(12, 6))

        for i, run in enumerate(self.history):
            plt.plot(run, alpha=0.3, color='blue', linewidth=0.5)

        max_len = max(len(run) for run in self.history)
        avg_convergence = []

        for epoch in range(max_len):
            epoch_values = [run[epoch] for run in self.history if epoch < len(run)]
            avg_convergence.append(np.mean(epoch_values))

        plt.plot(avg_convergence, color='red', linewidth=2, label='Mean Convergence')

        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Convergence analysis (10 Run)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class Sentinel2SegmentationProblem:

    def __init__(self, band_stack: np.ndarray, total_epochs: int,
                 n_thresholds=5, fusion_ratio=0.1):
        """
        Args:
            band_stack: Bands of size (H, W, C)
            total_epochs: Total number of iterations
            n_thresholds: Number of thresholds
            fusion_ratio: Ratio of iterations allocated to
        """
        self.band_stack = band_stack
        self.n_bands = band_stack.shape[-1]
        self.n_thresholds = n_thresholds

        self.fusion = DynamicBandFusion(band_stack, metric='entropy')
        self.otsu = OtsuMultiThreshold(n_thresholds=n_thresholds)

    def objective_function(self, solution: np.ndarray) -> float:

        weights = solution[:self.n_bands]
        thresholds = solution[self.n_bands:]
        exp_w = np.exp(weights - np.max(weights))
        weights = exp_w / (np.sum(exp_w) + 1e-10)


        hybrid_image = self.fusion.create_fusion_image(weights)

        # Correlation
        ref_image = np.mean(self.band_stack, axis=-1)
        corr = np.corrcoef(hybrid_image.ravel(), ref_image.ravel())[0, 1]
        corr = max(0.0, corr)
        corr_n = corr / (corr + 1.0)

        # 5. Otsu Between-Class Variance
        hist, bin_centers = self.otsu.compute_histogram(hybrid_image)
        otsu_var = self.otsu.between_class_variance(hist, bin_centers, thresholds)

        mu_global = np.sum(bin_centers * hist)
        total_var = np.sum(hist * (bin_centers - mu_global) ** 2)
        otsu_n = otsu_var / (total_var + 1e-10)

        segmented = self.otsu.apply_thresholds(hybrid_image, thresholds)
        labels = np.unique(segmented)

        # --- Inter-Class Contrast Proxy ---
        means = [hybrid_image[segmented == l].mean() for l in labels if np.any(segmented == l)]
        contrast = np.std(means) if len(means) > 1 else 0.0
        contrast_n = contrast / (contrast + 1.0)

        # --- Intra-Class Variance (F-measure proxy) ---
        intra_var = 0.0
        for l in labels:
            region = hybrid_image[segmented == l]
            if region.size > 1:
                intra_var += np.var(region)

        intra_var_n = 1.0 / (1.0 + intra_var)

        # JOINT FITNESS
        fitness = -(
            otsu_n
            * corr_n
            * contrast_n
            * intra_var_n
        )

        return fitness

def run_segmentation_pipeline(dataset_folder: str, scene_name: str,
                              hybrid_optimizer, epochs=100, n_runs=30):
    """
    Full segmentation pipeline

    Args:
        dataset_folder: Dataset root folder
        scene_name: Folder of the scene to be processed
        hybrid_optimizer: FinalHybridOptimizer instance
        epochs: Total number of iterations
        n_runs: Number of runs for stability testing
    """

    print("="*60)
    print("SENTINEL-2 MULTI-THRESHOLD SEGMENTATION")
    print("="*60)

    loader = Sentinel2DataLoader(dataset_folder)
    loader.load_scene(scene_name)
    loader.normalize_bands()
    loader.align_resolutions()
    band_stack = loader.get_band_stack()
    print(f"📊 Band Stack Shape: {band_stack.shape}")

    # 2. Problem
    problem = Sentinel2SegmentationProblem(
        band_stack=band_stack,
        total_epochs=epochs,
        n_thresholds=5,
        fusion_ratio=0.1
    )

    # Total dimension: Band + Threshold
    total_dims = band_stack.shape[-1] + 5

    print(f"\n[3/7] {n_runs} Bağımsız Run Başlatılıyor...")
    analyzer = ConvergenceAnalyzer()
    best_overall_fitness = float('inf')
    best_overall_solution = None

    # Her metrik için değerleri sakla
    all_metrics = {
        'Spectral-PSNR': [],
        'Spectral-SSIM': [],
        'F-measure': [],
        'Inter-Class Contrast': [],
        'Time': []
    }

    for run_id in range(n_runs):
        print(f"\nRun {run_id + 1}/{n_runs}")

        # Optimizasyon çalıştır
        from mealpy import FloatVar

        problem_dict = {
            "obj_func": problem.objective_function,
            "bounds": FloatVar(lb=[0.0]*total_dims, ub=[1.0]*total_dims),
            "minmax": "min",
            "log_to": None
        }

        start_time = time.perf_counter()
        result = hybrid_optimizer.solve(problem_dict)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        if hasattr(hybrid_optimizer, 'history'):
            analyzer.record_run(hybrid_optimizer.history.list_global_best_fit)

        best_solution = result.solution
        best_weights = best_solution[:problem.n_bands]
        best_thresholds = best_solution[problem.n_bands:]

        final_fusion_run = problem.fusion.create_fusion_image(best_weights)
        segmented_run = problem.otsu.apply_thresholds(final_fusion_run, best_thresholds)

        evaluator = SegmentationEvaluator()
        fidelity_run = evaluator.compute_spectral_fidelity(band_stack, final_fusion_run)
        quality_run = evaluator.compute_unsupervised_metrics(segmented_run, final_fusion_run)

        all_metrics['Spectral-PSNR'].append(fidelity_run['Spectral-PSNR'])
        all_metrics['Spectral-SSIM'].append(fidelity_run['Spectral-SSIM'])
        all_metrics['F-measure'].append(quality_run['F-measure'])
        all_metrics['Inter-Class Contrast'].append(quality_run['Inter-Class Contrast'])
        all_metrics['Time'].append(elapsed_ms)

        # Update global
        if result.target.fitness < best_overall_fitness:
            best_overall_fitness = result.target.fitness
            best_overall_solution = best_solution

        print(f"✅ Run {run_id + 1} Fitness: {result.target.fitness:.6f}")


    best_weights = best_overall_solution[:problem.n_bands]
    best_thresholds = best_overall_solution[problem.n_bands:]
    stats = analyzer.compute_statistics()

    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    print("\n📊 Metrik İstatistikleri ({} runs):".format(n_runs))
    print(f"  Spectral-PSNR: {np.mean(all_metrics['Spectral-PSNR']):.4f} ± {np.std(all_metrics['Spectral-PSNR']):.4f}")
    print(f"  Spectral-SSIM: {np.mean(all_metrics['Spectral-SSIM']):.4f} ± {np.std(all_metrics['Spectral-SSIM']):.4f}")
    print(f"  F-measure: {np.mean(all_metrics['F-measure']):.4f} ± {np.std(all_metrics['F-measure']):.4f}")
    print(f"  Inter-Class Contrast: {np.mean(all_metrics['Inter-Class Contrast']):.4f} ± {np.std(all_metrics['Inter-Class Contrast']):.4f}")
    print(f"  Time: {np.mean(all_metrics['Time']):.4f} ± {np.std(all_metrics['Time']):.4f}")


    final_fusion = problem.fusion.create_fusion_image(best_weights)
    segmented = problem.otsu.apply_thresholds(final_fusion, best_thresholds)

    evaluator = SegmentationEvaluator()
    fidelity = evaluator.compute_spectral_fidelity(band_stack, final_fusion)
    quality = evaluator.compute_unsupervised_metrics(segmented, final_fusion)

    avg_time = np.mean(all_metrics['Time'])
    metrics = {**fidelity, **quality, 'Time': avg_time}

    print("\n📈 En İyi Çözüm Değerlendirme Sonuçları:")
    print(f"  Spectral-PSNR: {metrics['Spectral-PSNR']:.4f}")
    print(f"  Spectral-SSIM: {metrics['Spectral-SSIM']:.4f}")
    print(f"  F-measure: {metrics['F-measure']:.4f}")
    print(f"  Inter-Class Contrast: {metrics['Inter-Class Contrast']:.4f}")
    print(f"  Time (ms): {metrics['Time']:.2f}")


    histogram_analyzer = HistogramAnalyzer()
    histogram_analyzer.plot_segmentation_histogram(
        fusion_image=final_fusion,
        segmented=segmented,
        thresholds=best_thresholds,
        save_path='histogram_analysis.png'
    )

    print("\nConvergence curve graph creating")
    analyzer.plot_convergence(save_path='convergence_analysis.png')

    print("\n" + "="*60)
    print("DONE")
    print("="*60)

    return {
        'segmented_image': segmented,
        'fusion_image': final_fusion,
        'best_thresholds': best_thresholds,
        'best_weights': best_weights,
        'metrics': metrics,
        'metrics_std': {
            'Spectral-PSNR_std': np.std(all_metrics['Spectral-PSNR']),
            'Spectral-SSIM_std': np.std(all_metrics['Spectral-SSIM']),
            'F-measure_std': np.std(all_metrics['F-measure']),
            'Inter-Class Contrast_std': np.std(all_metrics['Inter-Class Contrast'])
        },
        'statistics': stats,
        'band_stack': band_stack
    }

def batch_process_dataset(dataset_folder: str, hybrid_optimizer_class,
                         epochs=100, pop_size=30, switch_ratio=0.5,
                         n_runs=10, save_results=True):
    """
    Process all images in the dataset folder

    Args:
        dataset_folder: Dataset
        hybrid_optimizer_class: FinalHybridOptimizer class
        epochs: Number of iterations per image
        pop_size: Population size
        switch_ratio: GA/RUN transition ratio
        n_runs: Number of runs per image
        save_results: Save results (True/False)

    Returns:
        dict: Results for all images
    """

    print("="*70)
    print("Starting")
    print("="*70)

    # Dataset tarama
    loader = Sentinel2DataLoader(dataset_folder)
    image_folders = loader.available_images

    if not image_folders:
        raise ValueError("Images are not found in dataset folder")

    print(f"\n {len(image_folders)} \n")

    all_results = {}

    for idx, img_folder in enumerate(image_folders, 1):
        print(f"\n{'='*70}")
        print(f" [{idx}/{len(image_folders)}]: {img_folder}")
        print(f"{'='*70}\n")

        try:
            optimizer = hybrid_optimizer_class(
                epoch=epochs,
                pop_size=pop_size,
                switch_ratio=switch_ratio
            )

            results = run_segmentation_pipeline(
              dataset_folder=dataset_folder,
              scene_name=img_folder,  # ← image_folder yerine scene_name yazılmalı
              hybrid_optimizer=optimizer,
              epochs=epochs,
              n_runs=n_runs
          )

            all_results[img_folder] = results

            # Save results
            if save_results:
                save_path = Path(dataset_folder) / f"results_ga-{img_folder}"
                save_path.mkdir(exist_ok=True)

                # Save images
                plt.imsave(save_path / 'fusion.png',
                          results['fusion_image'], cmap='gray')
                plt.imsave(save_path / 'segmented.png',
                          results['segmented_image'], cmap='tab10')

                # Save the metrics
                import json
                metrics_dict = {
                    'metrics': results['metrics'],
                    'statistics': results['statistics'],
                    'thresholds': results['best_thresholds'].tolist(),
                    'weights': results['best_weights'].tolist()
                }

                with open(save_path / 'metrics.json', 'w') as f:
                    json.dump(metrics_dict, f, indent=4)

                print(f"\nResults are saved{save_path}")

        except Exception as e:
            print(f"\n ERROR ({img_folder}): {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[img_folder] = {'error': str(e)}
            continue

    print("\n" + "="*70)

    print("="*70)

    successful = sum(1 for r in all_results.values() if 'error' not in r)
    print(f"\n {successful}/{len(image_folders)}")

    if successful > 0:

        avg_contrast = np.mean([r['metrics']['Inter-Class Contrast']
                               for r in all_results.values() if 'error' not in r])

        std_contrast = np.std([r['metrics']['Inter-Class Contrast']
                               for r in all_results.values() if 'error' not in r])


        avg_f_measure = np.mean([r['metrics']['F-measure']
                                for r in all_results.values() if 'error' not in r])

        std_f_measure = np.std([r['metrics']['F-measure']
                                for r in all_results.values() if 'error' not in r])

        avg_psnr = np.mean([r['metrics']['Spectral-PSNR']
                           for r in all_results.values() if 'error' not in r])

        std_psnr = np.std([r['metrics']['Spectral-PSNR']
                           for r in all_results.values() if 'error' not in r])

        avg_ssim = np.mean([r['metrics']['Spectral-SSIM']
                           for r in all_results.values() if 'error' not in r])
        std_ssim = np.std([r['metrics']['Spectral-SSIM']
                           for r in all_results.values() if 'error' not in r])

        avg_time = np.mean([r['metrics']['Time']
                           for r in all_results.values() if 'error' not in r])
        std_time = np.std([r['metrics']['Time']
                           for r in all_results.values() if 'error' not in r])

        print(f"   Ortalama Inter-Class Contrast: {avg_contrast:.4f} ± {std_contrast:.4f} ")
        print(f"   Ortalama F-measure: {avg_f_measure:.4f} ± {std_f_measure:.4f}")
        print(f"   Ortalama Spectral-PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        print(f"   Ortalama Spectral-SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"   Ortalama Time: {avg_time:.4f} ± {std_time:.4f}")

    return all_results

all_results = batch_process_dataset(
    dataset_folder="..........................",
    hybrid_optimizer_class=FinalHybridOptimizer,
    epochs=100,
    pop_size=50,
    switch_ratio=0.5,
    n_runs=10,
    save_results=True
)


for image_name, results in all_results.items():
    if 'error' not in results:
        print(f"{image_name}: Spectral-PSNR = {results['metrics']['Spectral-PSNR']:.2f}")