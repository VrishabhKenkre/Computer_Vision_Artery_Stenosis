# F25_CVE_Denver – Automated Coronary Artery Stenosis Detection

This repository contains the code and experiments for our **F25 Computer Vision (CVE) project** on **Automated Coronary Artery Stenosis Detection**.

The goal of this project is to build an **automated, clinically meaningful stenosis-detection pipeline** that:

- Segments **coronary arteries** from raw X-ray angiograms using robust preprocessing + vesselness filtering
- Extracts accurate **vessel centerlines** and **local vessel diameters**
- Computes **Percent Diameter Stenosis (%DS)** along the vessel using validated reference-diameter methods
- Detects **significant narrowing** using a combined **geometry + intensity** criterion
- Generates rich **visual overlays**, **stenosis markers**, **diameter profiles**, **intensity profiles**, and **structured CSV outputs**
- Supports **scalable batch processing** for large datasets
- Provides a foundation for future **multi-vessel analysis**, **GPU acceleration**, and **clinical workflow integration**


> Core work and full pipeline are currently developed in main.

---

## 1. Project Overview


This project implements an automated pipeline to:

1. Preprocess raw angiogram images (contrast enhancement, noise reduction, field-of-view masking)
2. Perform **advanced vessel segmentation** using Frangi vesselness + morphological ops
3. Extract the **vessel skeleton** and estimate **diameter** along the centerline
4. Compute **reference diameter profiles** and **%DS** along the vessel
5. Detect and rank significant stenoses
6. Generate **PNG overlays, diameter profile plots, and CSV summaries**, both per-image and for entire datasets

---

## 2. Key Features

- **1. Advanced Vessel Preprocessing**
    The pipeline prepares raw angiogram frames for analysis through multiple enhancement steps, creating a stable foundation for segmentation:
    * **CLAHE contrast enhancement** improves visibility of faint vessels.
    * **Gaussian smoothing** reduces noise while preserving vascular edges.
    * **Field-of-view masking** removes irrelevant borders and artifacts.
    * **Black-hat morphology** suppresses low-frequency background illumination.

- **2. Robust Vessel Segmentation**
    Segmentation combines structural and intensity cues to isolate lumen regions:
    * Employs the **Frangi vesselness filter** to enhance tubular structures using Hessian eigenvalues. Vesselness $V(σ)$ is computed as:
        $$V = \exp\left(-\frac{R_B^2}{2\beta^2}\right)\left(1 - \exp\left(-\frac{S^2}{2\gamma^2}\right)\right)$$
    * **Otsu thresholding** applied to the black-hat image isolates candidate lumen regions.
    * Performs **Morphological cleanup** (closing $\rightarrow$ opening $\rightarrow$ hole filling $\rightarrow$ remove small fragments) to ensure a single clean mask.

- **4. Skeletonization & Main Vessel Centerline Extraction**
    This stage is one of the **core technical contributions** of the pipeline.
    *Steps:
    * Apply **Medial Axis Transform** → get binary skeleton and Euclidean distance map.  
    * Convert skeleton pixels into a **graph**:
      - Each skeleton pixel = node  
      - 8-connected neighborhood → weighted edges  
    * Extract **largest connected component (LCC)**  
    * Run **two-phase Dijkstra sweep** to estimate the **longest path**, which approximates the true arterial centerline.  
    * Reorder skeleton coordinates along this path into a **1D sequence** so the vessel can be analyzed like a signal.

- **4. Diameter Measurement (Geometry-based)**
    Using the ordered centerline, the pipeline computes an edge-based diameter:
    * At each point, a **local normal direction** is computed.
    * The diameter $D(i)$ is the distance between the two boundary hits along the normal, defined as:
        $$D(i) = | P_{\text{pos}} - P_{\text{neg}} |$$

- **5. Rolling Reference Diameter & Stenosis Quantification**
    To detect stenosis, a proximal reference is computed using a sliding window:
    * The **Reference Diameter ($D_{\text{ref}}$)** is calculated using the P90 metric over a sliding window ($W$):
        $$D_{\text{ref}}(i) = P_{90}(D(i-W: i+W))$$
    * The **Percent Diameter Stenosis (%DS)** is then calculated as:
        $$\%DS(i) = \left(1 - \frac{D(i)}{D_{\text{ref}}(i) + \varepsilon}\right) \times 100\%$$
    * A region is considered stenotic when %DS exceeds a clinical threshold (e.g., 50%). Lesions are expanded until %DS drops below this threshold.

- **6. Intensity-Based Stenosis Validation**
    This dual-evidence system increases reliability by validating geometry drops with contrast changes:
    * **Trace grayscale intensity** $I(i)$ along the skeleton.
    * Compute a rolling **P90 reference intensity** ($I_{\text{ref}}$).
    * Compute **percent intensity drop** ($\%ID$) and require that a stenosis is declared only when **both geometry shows a diameter drop** and **intensity shows a contrast drop**.

- **7. Clinical-Style Outputs**
    The pipeline automatically generates visualization and data outputs:
    * **Overlay images** with centerline, stenosis markers, caliper lines, and %DS labels.
    * **Diameter profile plots** and **Intensity profile plots**.
    * **Per-image CSV output** (diameter, reference, %DS, intensity metrics).
    * **Master summary CSV** detailing max %DS, counts of severe stenosis points, and lesion coordinates.

- **8. Batch Processing for Full Datasets**
    The system supports efficient analysis of large data collections:
    * Features **recursive folder scanning** and **multi-threaded execution**.
    * Enables **automatic generation** of per-image output folders and **dataset-level summary tables**.

---

## 3. Repository & Primary Workstreams

### Default Branch

- **`main`**
  - Minimal landing branch
  - Placeholder `README.md`
  - Meant to be updated with the consolidated project description (this file)

#### **Vrishabh — Lead Developer, Core Algorithm Architect**
- Designed and implemented the **full integrated pipeline**  
- Implemented **skeleton graph extraction**, **longest-path centerline**, and **per-pixel ordered artery profiling**  
- Developed **geometric stenosis detection**, **intensity profiling**, and **hybrid DS+PID stenosis scoring**  
- Built the entire **overlay system**, including calipers, markers, %DS labels  
- Designed **batch system, CSV infrastructure, summaries, 80–90% stenosis reporting**  
- Integrated all teammate modules into a cohesive final architecture  
- Tuned parameters, optimized segmentation, and resolved all failure-case logic  

#### **Sangram — Supporting Preprocessing & Early DV% Logic**
- Implemented early **CLAHE preprocessing** module  
- Contributed initial **DS% (diameter stenosis) logic**  
- Helped tune luminance preprocessing steps  

#### **Rajiv — Pseudocolor Blockage Detection Prototype**
- Developed blockage detection using:
  - Pseudocolor mapping  
  - Blue-region masking  
  - Contour extraction  
- Integrated into the final pipeline as supplementary evidence  

#### **Shweta — Binarization Experiments (Gamma + Black-Hat)**
- Built interactive segmentation using:
  - Gamma correction slider  
  - Black-hat morphology  
  - Manual thresholding  
- Integrated into final pipeline as `--mask_mode` alternative segmentation path  
---

## 4. Installation

### 4.1. Clone the repository

```bash
git clone https://github.com/VrishabhKenkre/Computer_Vision_Artery_Stenosis.git
cd Computer_Vision_Artery_Stenosis

To run the whole folder of Images
python stenosis_centerline_batch.py 
    --in "D:\Work\CV\Project\Data" 
    --out "D:\Work\CV\Project\Out_all" 
    --px 0.22 
    --pattern "*.png" 
    --recursive 
    --workers 6 
    --debug_plot
```

## 5. Python Environment

Install dependencies (example using pip):
```bash
pip install numpy pandas opencv-python scikit-image scipy matplotlib networkx
```
## 6. Acknowledgements

This project was developed as part of the F25 Computer Vision (CVE) course.

Team : Sangram Sahoo, Shweta Iyer, Rajiv Joshi, Vrishabh Kenkre
