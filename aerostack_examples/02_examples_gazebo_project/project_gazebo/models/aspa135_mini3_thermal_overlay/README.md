# ASPA135 Mini3 Thermal Overlay

## Overview

This model provides a thermal overlay for the ASPA135 Mini3 terrain to enable thermal camera simulation in Gazebo. Since Gazebo doesn't currently support heat signatures on heightmaps, this model creates a topographic thermal mesh that follows the actual terrain contours and works with the thermal camera plugin.

## Author

Juan Sandino, j.sandino@qut.edu.au


## Concept

- **Topographic Mesh**: Uses elevation data from the DEM to create a mesh that follows terrain contours
- **High-Resolution Options**: Configurable mesh resolution (32×32 to 256×256 vertices) for detailed terrain features
- **Thermal Plugin Compatible**: Works with Gazebo thermal system plugin (unlike heightmaps)
- **No Collisions**: The mesh has no collision geometry to avoid interfering with physics
- **Automatic Alignment**: Mesh is automatically oriented and positioned to match the terrain
- **Thermal Plugin**: Uses the Gazebo thermal system plugin with the actual thermal texture

## Usage

To use this thermal overlay with the main terrain:

1. Load the main terrain model (`aspa135_mini3`)
2. Load this thermal overlay model (`aspa135_mini3_thermal_overlay`) at the same world position
3. The thermal overlay will provide heat signature data for thermal cameras
4. Configure thermal cameras with temperature range: 263K to 313K (-6.34°C to 23.5°C)

## Technical Details

- **Mesh Size**: 60.0 x 60.0 meters (matches terrain dimensions)
- **Mesh Detail**: Configurable resolution:
  - **Low**: 32×32 = 1,024 vertices, 1,922 faces
  - **Medium**: 64×64 = 4,096 vertices, 7,938 faces (default)
  - **High**: 128×128 = 16,384 vertices, 32,258 faces
  - **Ultra**: 256×256 = 65,536 vertices, 130,050 faces
  - **In this case**: 1025×1025 = 1,050,625 vertices, 2,101,250 faces
- **Position**: Positioned 5cm above terrain (0, 0, 0.05) to avoid z-fighting
- **Mesh Format**: COLLADA (.dae) file with proper material definitions
- **Thermal Texture**: `thermal_ASPA135_Mini3Pro_or_gr_UTM_ROI_gz.png`
- **Temperature Range**: -6.34°C to 23.5°C (266.81K to 296.65K) - Antarctic conditions
- **Resolution**: 3.0 (thermal plugin resolution parameter)
- **Elevation Data**: Read from DEM TIFF file with bilinear interpolation
- **Automatic Orientation**: 180° rotation correction applied automatically

## Mesh Generation

The topographic mesh is generated using `gazebo_create_thermal_mesh.py`:

```bash
# Generate medium resolution mesh (default)
python3 gazebo_create_thermal_mesh.py

# Generate high resolution mesh for detailed terrain features
python3 gazebo_create_thermal_mesh.py --resolution 128

# Generate ultra-high resolution (warning: may impact performance)
python3 gazebo_create_thermal_mesh.py --resolution 256
```

### Requirements
- Python 3 with GDAL and SciPy
- Conda environment: `rsmeta` (contact Juan Sandino for access)

## Files

- `model.sdf`: Main model definition with topographic thermal mesh
- `model.config`: Model configuration and metadata  
- `meshes/thermal_plane.dae`: COLLADA topographic mesh file
- `generate_thermal_mesh.py`: Script to generate topographic thermal mesh from DEM
- `materials/textures/thermal_ASPA135_Mini3Pro_or_gr_UTM_ROI_gz.png`: Thermal texture

## Repository Information

**Important**: The complete thermal overlay functionality requires access to the `rs_meta` repository which contains the necessary conda environment and dependencies.

**Contact**: Juan Sandino to access the `rs_meta` repository and obtain the required environment setup.

## Features

- **Real Topography**: Mesh follows actual terrain elevation from DEM data
- **High Detail**: Captures rocks, ridges, and sharp terrain features
- **No Manual Adjustment**: Automatic orientation and positioning
- **Performance Scalable**: Choose resolution based on simulation requirements
- **Antarctic Realistic**: Temperature ranges suitable for polar research scenarios
- **Thermal Camera Ready**: Compatible with Gazebo thermal sensor system

## Notes

- The mesh uses double-sided rendering to ensure visibility from any angle
- Only thermal cameras will detect the heat signature from this overlay
- The model is static and has no physics simulation
- Higher resolution meshes provide better terrain detail but may impact performance
- Mesh automatically aligns with terrain - no manual rotation needed 