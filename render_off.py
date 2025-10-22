import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Path to your OFF file
file_path = "visual_result/dvmatcher_scape_r_std/deform_0.off"

# Load the mesh
mesh = trimesh.load(file_path, process=False)
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh.dump(concatenate=True)

print(mesh)

vertices = mesh.vertices
faces = mesh.faces

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

if len(faces) > 0:
    # Surface rendering
    mesh_collection = Poly3DCollection(vertices[faces], alpha=0.7,
                                       facecolor="lightblue", edgecolor="k")
    ax.add_collection3d(mesh_collection)
else:
    # Point cloud rendering
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               s=1, c="blue", alpha=0.6)

# Set axes limits
ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

plt.tight_layout()
plt.savefig("rendered_model.png", dpi=300)
print("Rendered image saved as rendered_model.png")
