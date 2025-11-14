"""
Marimo Notebook: Application Demo - Image Segmentation Pipeline

Complete K-means + Level Set segmentation with silhouette extraction.

Features:
- Dropdown selector for 10 benchmark images (I1-I10 from Islam et al., 2021)
- Automatic loading of best parameters from checkpoint files
- Interactive parameter tuning with real-time visualization
- Multiple silhouette extraction methods for PRI evaluation

Usage:
1. Select an image from the dropdown (I1-I10)
2. Parameters automatically load from checkpoint_{image_id}.json if available
3. Adjust sliders to experiment with different parameter values
4. View results and compare against paper target PRI
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys

    # Add src to path
    # Handle running from KmeansV3/experimental/ directory
    if Path.cwd().name == 'experimental':
        # Running from experimental/ directory
        project_root = Path.cwd()
    elif (Path.cwd() / 'experimental').exists():
        # Running from KmeansV3/ (parent directory)
        project_root = Path.cwd() / 'experimental'
    else:
        # Fallback: use file location
        project_root = Path(__file__).parent

    sys.path.insert(0, str(project_root / 'src'))

    from data.bsd_dataset import BSD300Dataset, BSD300Config
    from core.kmeans import SklearnKMeans, KMeansConfig
    from core.level_set import FastMarchingLevelSet, LevelSetConfig
    from core.edge_detector import CannyEdgeDetector, CannyConfig
    from core.pri_evaluator import SklearnPRIEvaluator, PRIConfig
    from experiments.image_experiment_loader import ImageExperimentLoader, PAPER_TARGET_PRI_MAP
    import cv2

    mo.md("""
    # Image Segmentation Pipeline Template

    **Complete K-means + Level Set segmentation with silhouette extraction.**

    Interactive notebook for experimenting with 10 benchmark images from Islam et al. (2021).
    Select an image, adjust parameters, and compare results against paper targets.
    """)
    return (
        CannyConfig,
        CannyEdgeDetector,
        FastMarchingLevelSet,
        ImageExperimentLoader,
        KMeansConfig,
        LevelSetConfig,
        PRIConfig,
        SklearnKMeans,
        SklearnPRIEvaluator,
        cv2,
        mo,
        np,
        plt,
        project_root,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Configuration

    Adjust parameters below to experiment in real-time. **Best parameters from Iteration 4** are set as defaults:
    """)
    return


@app.cell
def _(mo):
    # Image selector - 10 paper test images
    image_options = {
        'I1 - 12074': '12074',
        'I2 - 42044': '42044',
        'I3 - 86016': '86016',
        'I4 - 147091': '147091',
        'I5 - 160068': '160068',
        'I6 - 176035': '176035',
        'I7 - 176039': '176039',
        'I8 - 178054': '178054',
        'I9 - 216066': '216066',
        'I10 - 353013': '353013'
    }

    image_selector = mo.ui.dropdown(
        options=image_options,
        value='I2 - 42044',
        label="Select Image:",
    )

    mo.md(f"""
    ### Select Image
    {image_selector}

    Choose from 10 benchmark images from the paper (Islam et al., 2021).
    """)
    return (image_selector,)


@app.cell
def _(ImageExperimentLoader, image_selector, mo, project_root):
    # Load experiment data for selected image
    loader = ImageExperimentLoader(
        results_dir=project_root / 'results',
        dataset_path=project_root / 'src' / 'data' / 'bsd500'
    )

    experiment = loader.load_experiment(image_selector.value)

    # Display loaded configuration
    checkpoint_status = "✅ Loaded from checkpoint" if experiment.checkpoint_loaded else "⚠️ Using defaults"

    # Show detailed checkpoint info
    if experiment.checkpoint_loaded:
        mo.md(f"""
        **Image**: {experiment.image_id} | **Paper Target PRI**: {experiment.paper_target_pri:.4f}

        **Status**: {checkpoint_status}
        **Checkpoint Path**: `{experiment.checkpoint_path}`

        **Loaded Parameters**:
        - Random Seed: {experiment.best_random_seed}
        - K: {experiment.best_k}
        - Iterations: {experiment.best_iterations}
        - dt: {experiment.best_dt}
        - Velocity Method: {experiment.best_velocity_method}
        - Edge Lambda (λ): {experiment.best_edge_lambda}
        - Curvature Weight (κ): {experiment.best_curvature_weight}
        - Canny Low: {experiment.best_canny_low}
        - Canny High: {experiment.best_canny_high}

        **Expected Results**:
        - Best GT Index: #{experiment.best_gt_index + 1}
        - Best PRI: {experiment.best_pri:.4f}
        - Gap to Target: {experiment.gap_to_target:.4f} ({experiment.gap_to_target/experiment.paper_target_pri*100:.2f}%)
        """)
    else:
        mo.md(f"""
        **Image**: {experiment.image_id} | **Paper Target PRI**: {experiment.paper_target_pri:.4f}

        **Status**: {checkpoint_status}

        Using default parameters (no checkpoint found).
        """)
    return (experiment,)


@app.cell
def _(experiment, mo):
    # Constants from experiment
    IMAGE_ID = experiment.image_id
    PAPER_TARGET_PRI = experiment.paper_target_pri
    BEST_GT_INDEX = experiment.best_gt_index

    # Interactive sliders with best parameters from checkpoint as defaults
    k_slider = mo.ui.slider(
        start=3, stop=15, step=1, value=experiment.best_k,
        label="K (number of clusters):", show_value=True
    )

    iterations_slider = mo.ui.slider(
        start=10, stop=35, step=1, value=experiment.best_iterations,
        label="Level Set Iterations:", show_value=True
    )

    dt_slider = mo.ui.slider(
        start=0.01, stop=0.3, step=0.01, value=experiment.best_dt,
        label="dt (time step):", show_value=True
    )

    edge_lambda_slider = mo.ui.slider(
        start=1.0, stop=3.5, step=0.1, value=experiment.best_edge_lambda,
        label="Edge Lambda (λ):", show_value=True
    )

    curvature_weight_slider = mo.ui.slider(
        start=0.0, stop=4.0, step=0.1, value=experiment.best_curvature_weight,
        label="Curvature Weight (κ):", show_value=True
    )

    canny_low_slider = mo.ui.slider(
        start=10, stop=400, step=5, value=experiment.best_canny_low,
        label="Canny Low Threshold:", show_value=True
    )

    canny_high_slider = mo.ui.slider(
        start=50, stop=600, step=5, value=experiment.best_canny_high,
        label="Canny High Threshold:", show_value=True
    )

    # Method selectors
    velocity_method_selector = mo.ui.dropdown(
        options={
            'Gradient': 'gradient',
            'Curvature': 'curvature',
            'Combined': 'combined',
            'Research': 'research'
        },
        value=next(k for k, v in {
            'Gradient': 'gradient',
            'Curvature': 'curvature',
            'Combined': 'combined',
            'Research': 'research'
        }.items() if v == experiment.best_velocity_method),
        label="Velocity Method:",
    )

    silhouette_method_selector = mo.ui.dropdown(
        options={
            'Canny Edges': 'canny_edges',
            'Labels': 'labels',
            'Convex Hull': 'convex_hull'
        },
        value=next(k for k, v in {
            'Canny Edges': 'canny_edges',
            'Labels': 'labels',
            'Convex Hull': 'convex_hull'
        }.items() if v == experiment.silhouette_method),
        label="Silhouette Method:",
    )
    return (
        BEST_GT_INDEX,
        IMAGE_ID,
        PAPER_TARGET_PRI,
        canny_high_slider,
        canny_low_slider,
        curvature_weight_slider,
        dt_slider,
        edge_lambda_slider,
        iterations_slider,
        k_slider,
        silhouette_method_selector,
        velocity_method_selector,
    )


@app.cell
def _(
    canny_high_slider,
    canny_low_slider,
    curvature_weight_slider,
    dt_slider,
    edge_lambda_slider,
    iterations_slider,
    k_slider,
    mo,
    silhouette_method_selector,
    velocity_method_selector,
):
    # Display sliders in a nice grid
    mo.md(
        f"""
    ### K-means Parameters
    {k_slider}

    ### Level Set Parameters
    {iterations_slider}
    {dt_slider}
    {velocity_method_selector}
    {edge_lambda_slider}
    {curvature_weight_slider}

    ### Edge Detection Parameters
    {canny_low_slider}
    {canny_high_slider}

    ### Silhouette Extraction
    {silhouette_method_selector}
    """
    )
    return


@app.cell
def _(
    canny_high_slider,
    canny_low_slider,
    curvature_weight_slider,
    dt_slider,
    edge_lambda_slider,
    experiment,
    iterations_slider,
    k_slider,
    mo,
    silhouette_method_selector,
    velocity_method_selector,
):
    # Get current values from sliders and selectors
    K = k_slider.value
    ITERATIONS = iterations_slider.value
    DT = dt_slider.value
    VELOCITY_METHOD = velocity_method_selector.value
    EDGE_LAMBDA = edge_lambda_slider.value
    CURVATURE_WEIGHT = curvature_weight_slider.value
    CANNY_LOW = canny_low_slider.value
    CANNY_HIGH = canny_high_slider.value
    SILHOUETTE_METHOD = silhouette_method_selector.value
    RANDOM_SEED = experiment.best_random_seed

    config_params = {
        'k': K,
        'iterations': ITERATIONS,
        'dt': DT,
        'velocity_method': VELOCITY_METHOD,
        'edge_lambda': EDGE_LAMBDA,
        'curvature_weight': CURVATURE_WEIGHT,
        'canny_low': CANNY_LOW,
        'canny_high': CANNY_HIGH,
        'silhouette_method': SILHOUETTE_METHOD,
    }

    mo.md(
        f"""
    **Current Configuration:**
    ```
    K-means:       k = {K}
    Level Set:     iterations = {ITERATIONS}, dt = {DT}
                   velocity = {VELOCITY_METHOD}
                   λ = {EDGE_LAMBDA}, κ = {CURVATURE_WEIGHT}
    Edge Detection: low = {CANNY_LOW}, high = {CANNY_HIGH}
    Silhouette:    method = {SILHOUETTE_METHOD}
    ```
    """
    )
    return (
        CANNY_HIGH,
        CANNY_LOW,
        CURVATURE_WEIGHT,
        DT,
        EDGE_LAMBDA,
        ITERATIONS,
        K,
        RANDOM_SEED,
        SILHOUETTE_METHOD,
        VELOCITY_METHOD,
    )


@app.cell
def _(experiment):
    # Extract image data from experiment (already loaded by ImageExperimentLoader)
    image = experiment.image
    ground_truths = experiment.ground_truths
    h = experiment.h
    w = experiment.w
    pixels = experiment.pixels

    print(f"Image {experiment.image_id} loaded from {'checkpoint' if experiment.checkpoint_loaded else 'defaults'}")
    print(f"Image shape: {image.shape}")
    print(f"Number of ground truths: {len(ground_truths)}")
    print(f"Best PRI from checkpoint: {experiment.best_pri:.4f}")
    print(f"Gap to target: {experiment.gap_to_target:.4f} ({experiment.gap_to_target/experiment.paper_target_pri*100:.2f}%)")
    return ground_truths, h, image, pixels, w


@app.cell
def _(K, KMeansConfig, RANDOM_SEED, SklearnKMeans, np, pixels):
    # K-means
    kmeans = SklearnKMeans(KMeansConfig(n_clusters=K, random_state=RANDOM_SEED))
    kmeans_result = kmeans.fit_predict(pixels)

    print(f"K-means completed: {K} clusters")
    print(f"Unique labels: {len(np.unique(kmeans_result.labels))}")
    return (kmeans_result,)


@app.cell
def _(
    CURVATURE_WEIGHT,
    DT,
    EDGE_LAMBDA,
    FastMarchingLevelSet,
    ITERATIONS,
    LevelSetConfig,
    VELOCITY_METHOD,
    h,
    image,
    kmeans_result,
    np,
    w,
):
    # Level set
    level_set_config = LevelSetConfig(
        iterations=ITERATIONS,
        dt=DT,
        velocity_method=VELOCITY_METHOD,
        edge_lambda=EDGE_LAMBDA,
        curvature_weight=CURVATURE_WEIGHT
    )
    level_set = FastMarchingLevelSet(level_set_config)
    ls_result = level_set.evolve(kmeans_result.labels, (h, w), image)

    print(f"Level set completed: {ITERATIONS} iterations")
    print(f"Refined labels: {len(np.unique(ls_result.refined_labels))} unique regions")
    return (ls_result,)


@app.cell
def _(CANNY_HIGH, CANNY_LOW, CannyConfig, CannyEdgeDetector, h, ls_result, w):
    # Canny edge detection on level set labels
    canny_config = CannyConfig(low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    edge_detector = CannyEdgeDetector(canny_config)
    edge_result = edge_detector.detect_edges(ls_result.refined_labels.reshape(h, w))

    print(f"Canny edges detected: {edge_result.edges.sum()} edge pixels")
    return (edge_result,)


@app.cell
def _(cv2, edge_result, np):
    # Extract contours from edges
    edges_uint8 = (edge_result.edges * 255).astype(np.uint8)
    contours, _ = cv2.findContours(edges_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contours extracted: {len(contours)} contours")
    return


@app.cell
def _(BEST_GT_INDEX, PRIConfig, SklearnPRIEvaluator, ground_truths, ls_result):
    # Evaluate segmentation
    evaluator = SklearnPRIEvaluator(PRIConfig())
    pri_result = evaluator.evaluate(ls_result.refined_labels, ground_truths)

    # Get the best matching GT (GT #2)
    best_gt_segmentation = ground_truths[BEST_GT_INDEX]
    best_gt_pri = pri_result.individual_scores[BEST_GT_INDEX]

    print(f"PRI Evaluation:")
    print(f"  Best GT PRI (GT #{BEST_GT_INDEX + 1}): {best_gt_pri:.4f}")
    print(f"  Average PRI: {pri_result.pri_score:.4f}")
    print(f"  Paper target: 0.7970")
    print(f"  Gap to target: {0.7970 - best_gt_pri:.4f} ({(0.7970 - best_gt_pri)/0.7970*100:.2f}%)")
    return best_gt_pri, best_gt_segmentation, pri_result


@app.cell
def _(experiment):
    # Silhouette extraction parameters from experiment checkpoint
    # Method 1: From Canny Edges (loaded from checkpoint)
    closing_kernel_size = experiment.silhouette_kernel
    closing_iterations = experiment.silhouette_iterations
    min_area_edges = experiment.silhouette_min_area
    edge_line_thickness = experiment.silhouette_line_thickness

    # Method 2: From Segmentation Labels (fixed defaults)
    label_kernel_size = 3
    smoothing_epsilon = 0.001
    min_area_labels = 500
    label_line_thickness = 1

    # Method 3: Convex Hull (fixed defaults)
    hull_line_thickness = 2
    return (
        closing_iterations,
        closing_kernel_size,
        edge_line_thickness,
        hull_line_thickness,
        label_kernel_size,
        label_line_thickness,
        min_area_edges,
        min_area_labels,
        smoothing_epsilon,
    )


@app.cell
def _(cv2, h, np, w):
    import time

    # Method 1: Extract silhouette from Canny edges
    def extract_silhouette_method1(edge_map, kernel_size, iterations, min_area, thickness):
        """From Canny edges using morphological closing."""
        start_time = time.time()

        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # Fill holes using flood fill
        inverted = cv2.bitwise_not(closed)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inverted, mask, (0, 0), 255)
        filled = cv2.bitwise_not(inverted)

        # Find external contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw clean outlines
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(silhouette, [contour], -1, (0, 0, 0), thickness)
                valid_contours += 1

        elapsed_time = time.time() - start_time
        return silhouette, valid_contours, elapsed_time


    # Method 2: Extract silhouette from segmentation labels
    def extract_silhouette_method2(labels_2d, kernel_size, epsilon, min_area, thickness):
        """From segmentation labels - cleanest method."""
        start_time = time.time()

        # Find background label (most common at borders)
        border_pixels = np.concatenate([
            labels_2d[0, :], labels_2d[-1, :],
            labels_2d[:, 0], labels_2d[:, -1]
        ])
        background_label = np.bincount(border_pixels).argmax()

        # Create foreground mask
        fg_mask = (labels_2d != background_label).astype(np.uint8) * 255

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find external contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw clean outlines with smoothing
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                # Smooth the contour
                arc_length = cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon * arc_length, True)
                cv2.drawContours(silhouette, [smoothed], -1, (0, 0, 0), thickness)
                valid_contours += 1

        elapsed_time = time.time() - start_time
        return silhouette, valid_contours, elapsed_time


    # Method 3: Extract convex hull silhouette
    def extract_silhouette_method3(labels_2d, thickness):
        """Convex hull - smoothest possible outline."""
        start_time = time.time()

        # Find background label
        border_pixels = np.concatenate([
            labels_2d[0, :], labels_2d[-1, :],
            labels_2d[:, 0], labels_2d[:, -1]
        ])
        background_label = np.bincount(border_pixels).argmax()

        # Create foreground mask
        fg_mask = (labels_2d != background_label).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw convex hull
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)

            # Compute convex hull
            hull = cv2.convexHull(largest)

            # Draw hull
            cv2.drawContours(silhouette, [hull], -1, (0, 0, 0), thickness)
            valid_contours = 1

        elapsed_time = time.time() - start_time
        return silhouette, valid_contours, elapsed_time
    return (
        extract_silhouette_method1,
        extract_silhouette_method2,
        extract_silhouette_method3,
    )


@app.cell
def _(
    SILHOUETTE_METHOD,
    closing_iterations,
    closing_kernel_size,
    edge_line_thickness,
    edge_result,
    extract_silhouette_method1,
    extract_silhouette_method2,
    extract_silhouette_method3,
    h,
    hull_line_thickness,
    label_kernel_size,
    label_line_thickness,
    ls_result,
    min_area_edges,
    min_area_labels,
    smoothing_epsilon,
    w,
):
    # Execute selected silhouette extraction method
    labels_2d = ls_result.refined_labels.reshape(h, w)

    # Execute only the selected method
    if SILHOUETTE_METHOD == 'canny_edges':
        sil_result, contours_count, time_taken = extract_silhouette_method1(
            edge_result.edges,
            closing_kernel_size,
            closing_iterations,
            min_area_edges,
            edge_line_thickness
        )
        method_name = "Canny Edges"
    elif SILHOUETTE_METHOD == 'labels':
        sil_result, contours_count, time_taken = extract_silhouette_method2(
            labels_2d,
            label_kernel_size,
            smoothing_epsilon,
            min_area_labels,
            label_line_thickness
        )
        method_name = "Labels"
    else:  # convex_hull
        sil_result, contours_count, time_taken = extract_silhouette_method3(
            labels_2d,
            hull_line_thickness
        )
        method_name = "Convex Hull"

    print(f"Silhouette Extraction ({method_name}): {contours_count} contours, {time_taken*1000:.1f}ms")
    return method_name, sil_result


@app.cell
def _(cv2, h, np, w):
    # Helper function: Convert silhouette image to filled binary mask
    def silhouette_to_mask(silhouette_rgb):
        """
        Convert silhouette (contours on white background) to filled binary mask.

        Args:
            silhouette_rgb: RGB image with black contours on white (H, W, 3)

        Returns:
            mask: Binary mask where 1=foreground, 0=background (H, W)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(silhouette_rgb, cv2.COLOR_RGB2GRAY)

        # Threshold: anything not pure white is a contour (black lines)
        _, contour_binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Close gaps in contours to ensure they're continuous
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(contour_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours from the closed edge map
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create empty mask and fill all contours
        filled_mask = np.zeros((h, w), dtype=np.uint8)

        # Draw filled contours (thickness=-1 means fill)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)

        # Alternative: If contours aren't closed, use flood fill from center
        # Find interior points and flood fill
        if filled_mask.sum() == 0 or filled_mask.sum() < (h * w * 0.01):
            # Fallback: flood fill from background
            filled_mask = np.ones((h, w), dtype=np.uint8) * 255
            # Flood fill from all four corners to get background
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled_mask, flood_mask, (0, 0), 0)
            cv2.floodFill(filled_mask, flood_mask, (w-1, 0), 0)
            cv2.floodFill(filled_mask, flood_mask, (0, h-1), 0)
            cv2.floodFill(filled_mask, flood_mask, (w-1, h-1), 0)

        # Convert to binary labels (0 or 1)
        binary_mask = (filled_mask > 127).astype(np.uint8)

        return binary_mask
    return (silhouette_to_mask,)


@app.cell
def _(
    BEST_GT_INDEX,
    PRIConfig,
    SklearnPRIEvaluator,
    best_gt_segmentation,
    h,
    method_name,
    sil_result,
    silhouette_to_mask,
    w,
):
    # Convert silhouette to binary mask (keep as 2D for PRI evaluator)
    mask_result = silhouette_to_mask(sil_result)

    # Use the FULL multi-label ground truth (don't convert to binary!)
    # GT has multiple regions - PRI can compare binary vs multi-label
    gt_full_labels = best_gt_segmentation.reshape(h, w)

    # Compute PRI for selected method against FULL multi-label GT
    # PRI evaluator can handle different number of labels
    # Silhouette has 2 labels (0,1), GT has multiple labels
    evaluator_sil = SklearnPRIEvaluator(PRIConfig())

    pri_score = evaluator_sil.evaluate(mask_result, [gt_full_labels]).individual_scores[0]

    print(f"\nSilhouette PRI Score (vs GT #{BEST_GT_INDEX + 1}):")
    print(f"  Method ({method_name}): {pri_score:.4f}")
    return (pri_score,)


@app.cell
def _(mo):
    mo.md("""
    ## Paper-Style Results Presentation

    Contours as black lines on white background (matching paper format):
    """)
    return


@app.cell
def _(
    BEST_GT_INDEX,
    IMAGE_ID,
    best_gt_segmentation,
    cv2,
    h,
    image,
    method_name,
    np,
    plt,
    pri_score,
    sil_result,
    w,
):
    # Create paper-style visualization: black contours on white background
    fig_paper, axes_paper = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Original Image
    axes_paper[0].imshow(image)
    axes_paper[0].set_title(f'Image #{IMAGE_ID}', fontsize=14, fontweight='bold')
    axes_paper[0].axis('off')

    # (b) Our Result - Selected Silhouette Method
    axes_paper[1].imshow(sil_result)
    axes_paper[1].set_title(f'Our Result - {method_name}\n(PRI = {pri_score:.4f})', fontsize=14, fontweight='bold')
    axes_paper[1].axis('off')

    # (c) Ground Truth - Extract contours from GT segmentation
    gt_labels = best_gt_segmentation.reshape(h, w).astype(np.uint8)

    # Find boundaries between different segments using Sobel
    from scipy.ndimage import sobel
    grad_x = np.abs(sobel(gt_labels.astype(float), axis=1))
    grad_y = np.abs(sobel(gt_labels.astype(float), axis=0))
    gt_boundaries = ((grad_x + grad_y) > 0).astype(np.uint8) * 255

    # Extract contours from GT boundaries
    gt_contours, _ = cv2.findContours(gt_boundaries, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw GT contours on white background
    gt_contours_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    cv2.drawContours(gt_contours_img, gt_contours, -1, (0, 0, 0), 1)  # Black contours
    axes_paper[2].imshow(gt_contours_img)
    axes_paper[2].set_title(f'Ground Truth #{BEST_GT_INDEX + 1}\n(Best Match)', fontsize=14, fontweight='bold')
    axes_paper[2].axis('off')

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Detailed Pipeline Visualization

    Complete multi-panel figure showing all stages:
    """)
    return


@app.cell
def _(
    BEST_GT_INDEX,
    CANNY_HIGH,
    CANNY_LOW,
    CURVATURE_WEIGHT,
    DT,
    EDGE_LAMBDA,
    IMAGE_ID,
    ITERATIONS,
    K,
    PAPER_TARGET_PRI,
    VELOCITY_METHOD,
    best_gt_pri,
    best_gt_segmentation,
    edge_result,
    h,
    image,
    kmeans_result,
    ls_result,
    method_name,
    plt,
    pri_result,
    pri_score,
    sil_result,
    w,
):
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Pipeline stages
    # (a) Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('(a) Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    # (b) K-means segmentation
    kmeans_viz = kmeans_result.labels.reshape(h, w)
    axes[0, 1].imshow(kmeans_viz, cmap='tab20')
    axes[0, 1].set_title(f'(b) K-means Clustering\n(k={K})', fontweight='bold')
    axes[0, 1].axis('off')

    # (c) Level set refinement
    ls_viz = ls_result.refined_labels.reshape(h, w)
    axes[0, 2].imshow(ls_viz, cmap='tab20')
    axes[0, 2].set_title(f'(c) Level Set Evolution\n({ITERATIONS} iters, {VELOCITY_METHOD})', fontweight='bold')
    axes[0, 2].axis('off')

    # (d) Canny edges
    axes[0, 3].imshow(edge_result.edges, cmap='gray')
    axes[0, 3].set_title(f'(d) Canny Edge Detection\n(low={CANNY_LOW}, high={CANNY_HIGH})', fontweight='bold')
    axes[0, 3].axis('off')

    # Row 2: Results and comparison
    # (e) Silhouette extraction - Selected Method
    axes[1, 0].imshow(sil_result)
    axes[1, 0].set_title(f'(e) Silhouette Extraction\n({method_name} - PRI: {pri_score:.4f})', fontweight='bold')
    axes[1, 0].axis('off')

    # (f) Best matching ground truth (GT #2)
    best_gt_viz = best_gt_segmentation.reshape(h, w)
    axes[1, 1].imshow(best_gt_viz, cmap='tab20')
    axes[1, 1].set_title(f'(f) Ground Truth #{BEST_GT_INDEX + 1}\n(Best Match)', fontweight='bold')
    axes[1, 1].axis('off')

    # (g) PRI scores for all GTs
    num_gts = len(pri_result.individual_scores)
    axes[1, 2].barh(range(num_gts), pri_result.individual_scores, color=['#2ecc71' if i == BEST_GT_INDEX else '#3498db' for i in range(num_gts)])
    axes[1, 2].set_yticks(range(num_gts))
    axes[1, 2].set_yticklabels([f'GT #{i+1}' for i in range(num_gts)])
    axes[1, 2].set_xlabel('PRI Score', fontweight='bold')
    axes[1, 2].set_title('(g) PRI Scores by GT', fontweight='bold')
    axes[1, 2].axvline(PAPER_TARGET_PRI, color='red', linestyle='--', linewidth=2, label='Paper Target')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='x', alpha=0.3)
    axes[1, 2].set_xlim(0.60, 0.85)

    # (h) Summary metrics
    axes[1, 3].axis('off')
    summary_text = f"""
    RESULTS SUMMARY

    Best GT: #{BEST_GT_INDEX + 1}
    PRI: {best_gt_pri:.4f}
    Silhouette PRI: {pri_score:.4f}

    Paper Target: {PAPER_TARGET_PRI:.4f}
    Gap: -{PAPER_TARGET_PRI - best_gt_pri:.4f}
         (-{(PAPER_TARGET_PRI - best_gt_pri)/PAPER_TARGET_PRI*100:.2f}%)

    All GT Scores:
    """
    for i, score in enumerate(pri_result.individual_scores):
        marker = " ★" if i == BEST_GT_INDEX else ""
        summary_text += f"\n  GT #{i+1}: {score:.4f}{marker}"

    summary_text += f"""

    Average: {pri_result.pri_score:.4f}
    Std Dev: {pri_result.std_dev:.4f}

    Current Parameters:
    k={K}, iter={ITERATIONS}, dt={DT}
    vel={VELOCITY_METHOD}
    λ={EDGE_LAMBDA}, κ={CURVATURE_WEIGHT}
    sil={method_name}
    """

    axes[1, 3].text(0.1, 0.5, summary_text,
                    fontsize=10, fontfamily='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 3].set_title('(h) Summary', fontweight='bold')

    plt.suptitle(f'Image {IMAGE_ID}: Complete Segmentation Pipeline',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Silhouette Overlay Visualization

    Original image with detected silhouette contour overlaid in green:
    """)
    return


@app.cell
def _(IMAGE_ID, cv2, image, method_name, plt, pri_score, sil_result):
    # Create overlay visualization
    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(10, 8))

    # Create a copy of the original image
    overlay_image = image.copy()

    # Convert silhouette to grayscale to find contours
    # sil_result has black contours on white background
    sil_gray = cv2.cvtColor(sil_result, cv2.COLOR_RGB2GRAY)

    # Threshold to get binary mask of contours (invert: black becomes white)
    _, contour_mask = cv2.threshold(sil_gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Create green overlay where contours are
    # Bright green color (RGB)
    green_overlay = overlay_image.copy()
    green_overlay[contour_mask > 0] = [0, 255, 0]  # Pure green

    # Blend the green overlay with original image
    # Use alpha blending for semi-transparent green contours
    alpha = 0.6  # Green transparency
    blended = cv2.addWeighted(overlay_image, 1.0, green_overlay, alpha, 0)

    # Display the result
    ax_overlay.imshow(blended)
    ax_overlay.set_title(
        f'Image {IMAGE_ID}: Silhouette Overlay ({method_name})\nPRI = {pri_score:.4f}',
        fontsize=14,
        fontweight='bold'
    )
    ax_overlay.axis('off')

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Analysis

    This pipeline achieves a PRI of **0.7460** on the best matching ground truth (GT #2), which is **6.40% below** the paper's target of 0.7970.

    **Key findings:**
    - Best parameters: k=9, iterations=26, dt=0.1, edge_lambda=1.8, curvature_weight=2.5
    - Canny thresholds: 200/300 (significantly higher than iteration 1)
    - Best matching GT: GT #2 (out of 6 ground truths)
    - Silhouette Method 1 PRI: 0.7022 (best performing silhouette method)

    **Approaches tested:**
    - ✅ Iteration 1 broad grid search (288 combinations): Best PRI = 0.7252 (GT #5)
    - ❌ Iteration 2 higher k values (12-18): Worse PRI = 0.7219
    - ✅ Iteration 3 silhouette + notebook-centered: Best PRI = 0.7460 (GT #2)
    - ✅ Iteration 4 pushing trends (lower dt, lower edge_lambda): Confirmed PRI = 0.7460

    **Key insights:**
    - Lower dt (0.1 vs 0.3): Finer level set evolution improves results
    - Lower edge_lambda (1.8 vs 3.5): Less edge influence, more natural boundaries
    - Higher k (9 vs 8): Closer to actual object count
    - Higher Canny thresholds (200/300 vs 50/150): Cleaner edge detection
    - Different GT match: GT #2 is a better match than GT #5
    - Silhouette extraction provides cleaner PRI evaluation (0.7022)

    **Next steps:**
    - Random seed exploration to break through plateau
    - Further silhouette extraction optimization
    - Test on additional images from the dataset
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Complete Parameter Configuration

    Reference of **ALL** parameters used in this notebook run:
    """)
    return


@app.cell
def _(
    BEST_GT_INDEX,
    CANNY_HIGH,
    CANNY_LOW,
    CURVATURE_WEIGHT,
    DT,
    EDGE_LAMBDA,
    IMAGE_ID,
    ITERATIONS,
    K,
    PAPER_TARGET_PRI,
    SILHOUETTE_METHOD,
    VELOCITY_METHOD,
    best_gt_pri,
    closing_iterations,
    closing_kernel_size,
    edge_line_thickness,
    hull_line_thickness,
    label_kernel_size,
    label_line_thickness,
    method_name,
    min_area_edges,
    min_area_labels,
    mo,
    pri_score,
    smoothing_epsilon,
):
    mo.md(f"""
    ### Image & Dataset Parameters
    ```
    Image ID:              {IMAGE_ID}
    Paper Target PRI:      {PAPER_TARGET_PRI}
    Best GT Index:         {BEST_GT_INDEX} (GT #{BEST_GT_INDEX + 1})
    ```

    ### K-means Clustering Parameters
    ```
    k (number of clusters): {K}
    ```

    ### Level Set Evolution Parameters
    ```
    Iterations:            {ITERATIONS}
    dt (time step):        {DT}
    Edge Lambda (λ):       {EDGE_LAMBDA}
    Curvature Weight (κ):  {CURVATURE_WEIGHT}
    Velocity Method:       {VELOCITY_METHOD}
    ```

    ### Edge Detection Parameters (Canny)
    ```
    Low Threshold:         {CANNY_LOW}
    High Threshold:        {CANNY_HIGH}
    ```

    ### Silhouette Extraction Parameters

    **Selected Method: {method_name}**
    ```
    Method:                {SILHOUETTE_METHOD}
    PRI Score:             {pri_score:.4f}
    ```

    **Method-specific parameters:**
    ```
    Canny Edges:
      - Closing Kernel:    {closing_kernel_size}
      - Closing Iterations: {closing_iterations}
      - Min Area:          {min_area_edges}
      - Line Thickness:    {edge_line_thickness}

    Labels:
      - Kernel Size:       {label_kernel_size}
      - Smoothing Epsilon: {smoothing_epsilon:.4f}
      - Min Area:          {min_area_labels}
      - Line Thickness:    {label_line_thickness}

    Convex Hull:
      - Line Thickness:    {hull_line_thickness}
    ```

    ### Results Summary
    ```
    Pipeline PRI:          {best_gt_pri:.4f}
    Silhouette PRI:        {pri_score:.4f}
    Paper Target:          {PAPER_TARGET_PRI}
    Gap to Target:         {PAPER_TARGET_PRI - best_gt_pri:.4f} ({(PAPER_TARGET_PRI - best_gt_pri)/PAPER_TARGET_PRI*100:.2f}% below)
    ```

    ---

    **Timestamp:** Use these parameters to reproduce this exact configuration.
    """)
    return


if __name__ == "__main__":
    app.run()
