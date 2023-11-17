import numpy as np
import open3d as o3d
from PIL import Image
import json

def parse_obj(filename):
    vertices = []
    normals = []
    uvs = []
    faces = []
    colors = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                data = [float(x) for x in line[2:].split()]
                vertices.append(data[:3])
                colors.append(data[3:])
            elif line.startswith('vn '):
                normals.append([float(x) for x in line[3:].split()])
            elif line.startswith('vt '):
                uvs.append([float(x) for x in line[3:].split()])
            elif line.startswith('f '):
                triangle = [x.split('/') for x in line.split()[1:]]
                triangle = [[int(x) for x in vertex] for vertex in triangle]
                faces.append(triangle)

    data = {}
    data['vertices']    = np.array(vertices)
    data['normals']     = np.array(normals)
    data['uvs']         = np.array(uvs)
    data['faces']       = np.array(faces)
    data['colors']      = np.array(colors)

    return data

def save_obj(filename, data):
    vertices = data.get('vertices', np.array([]))
    normals  = data.get('normals' , np.array([]))
    uvs      = data.get('uvs'     , np.array([]))
    faces    = data.get('faces'   , np.array([]))
    colors   = data.get('colors'  , np.array([]))

    with open(filename, 'w') as f:

        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n")

        for i in range(len(vertices)):
            vertex = vertices[i]
            normal = normals[i]
            color = colors[i] if len(colors) else ''

            f.write('v ')
            f.write(f"{' '.join(str(round(x, 2)) for x in vertex)}")
            f.write(' ')
            f.write(f"{' '.join(str(round(x, 6)) for x in color)}")
            f.write('\n')

            f.write('vn ')
            f.write(f"{' '.join(str(round(x, 6)) for x in normal)}")
            f.write('\n')

        for uv in uvs:
            f.write(f"vt {' '.join(str(round(x, 6)) for x in uv)}\n")

        for face in faces:
            indices = ' '.join(['/'.join(map(str, vertex)) for vertex in face])
            f.write(f"f {indices}\n")

def uvs_to_colors(data):
    uvs = data.get('uvs', np.array([]))

    rg = uvs
    b  = np.ones((len(uvs), 1))

    data['colors'] = np.hstack((rg, b))

def colors_to_uvs(data):
    colors = data.get('colors', np.array([]))

    data['uvs']    = colors[:, :2]
    data['colors'] = np.array([])

def mesh_to_data(mesh):
    # uvs is missed
    data = {}
    data['vertices']    = np.asarray(mesh.vertices)
    data['normals']     = np.asarray(mesh.vertex_normals)
    data['colors']      = np.asarray(mesh.vertex_colors)

    # FACE_NUM * 3     : [[0, 2, 5], ...]
    f = np.array(mesh.triangles)
    # FACE_NUM * 3 * 3 : [[[1, 1, 1], [3, 3, 3], [6, 6, 6]], ...]
    data['faces']  = (f + 1)[:, :, np.newaxis] * np.ones_like(f)[:, np.newaxis, :]

    return data

def simplify_obj(OBJ_INPUT, OBJ_SIMPLIFY, SIMPLIFY):
    data = parse_obj(OBJ_INPUT)

    # Open3D decimation can't deal with UV info, so let's save those info as vertices colors (it's a trick :P)
    uvs_to_colors(data)
    save_obj(OBJ_SIMPLIFY, data)

    # mesh decimation
    mesh = o3d.io.read_triangle_mesh(OBJ_SIMPLIFY)
    TRI_NUM = len(mesh.triangles) // SIMPLIFY
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=TRI_NUM)

    # Once Open3D finishes decimation, restore those uv info from vertices colors
    data = mesh_to_data(mesh_smp)
    colors_to_uvs(data)
    save_obj(OBJ_SIMPLIFY, data)

def re_index(data):
    data['faces'] -= 1
    selected_vertices = np.unique(data['faces'][:,:,0])

    # only leave the vertices used to form the faces
    data['vertices'] = data['vertices'][selected_vertices]
    data['normals']  = data['normals'][selected_vertices]
    data['uvs']      = data['uvs'][selected_vertices]
    data['colors']   = data['colors'][selected_vertices]

    # update face index
    vertex_mapping = { old_index: new_index for new_index, old_index in enumerate(selected_vertices) }
    data['faces'] = np.vectorize(lambda x: vertex_mapping.get(x, x))(data['faces'])
    data['faces'] += 1

def cut_obj(OBJ_SIMPLIFY, X_CENTER, OBJ_SIMPLIFY_LEFT, OBJ_SIMPLIFY_RIGHT):
    data = parse_obj(OBJ_SIMPLIFY)

    tri_z_values = data['vertices'][data['faces'][:,:,0] - 1, 2]
    centers = np.interp(tri_z_values, X_CENTER[:, 2], X_CENTER[:, 0])

    # number of vertices of each triangle located in the left side (0~3)
    tri_left_num = np.sum(data['vertices'][data['faces'][:,:,0] - 1, 0] < centers, axis=1)

    # make vertices align with the cutting edge (left & right triangle)
    right_edge_faces = data['faces'][tri_left_num == 1][:,:,0] - 1
    mask = data['vertices'][right_edge_faces, 0] < centers[tri_left_num == 1]
    mask[:, 0] = np.logical_or(mask[:, 0], np.logical_not(mask.any(axis=1))) # exclude [False, False, False]
    data['vertices'][right_edge_faces[mask], 0] = centers[tri_left_num == 1, np.argmax(mask, axis=1)]

    left_edge_faces = data['faces'][tri_left_num == 2][:,:,0] - 1
    mask = data['vertices'][left_edge_faces, 0] > centers[tri_left_num == 2]
    mask[:, 0] = np.logical_or(mask[:, 0], np.logical_not(mask.any(axis=1))) # exclude [False, False, False]
    data['vertices'][left_edge_faces[mask], 0] = centers[tri_left_num == 2, np.argmax(mask, axis=1)]

    # receive left & right surface
    left_data = data.copy()
    right_data = data.copy()

    left_data['faces'] = data['faces'][tri_left_num >= 2]
    right_data['faces'] = data['faces'][tri_left_num < 2]

    save_obj(OBJ_SIMPLIFY_LEFT, left_data)
    save_obj(OBJ_SIMPLIFY_RIGHT, right_data)

def cluster_obj(filename, SEGMENT_ID, SIMPLIFY, REVERSE, LEFT):
    data = parse_obj(filename)
    mesh = o3d.io.read_triangle_mesh(filename)

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    cluster_data_list = []
    cluster_uv_list = []
    for i in range(len(cluster_n_triangles)):
        cluster_data = data.copy()
        cluster_data['faces'] = data['faces'][triangle_clusters == i]

        if(cluster_data['faces'].shape[0] < 100): continue

        re_index(cluster_data)
        cluster_data_list.append(cluster_data)
        cluster_uv_list.append(np.mean(cluster_data['uvs'], axis=0))

    sorted_id = np.argsort(np.array(cluster_uv_list)[:, 0])
    sorted_id = sorted_id if not REVERSE else np.max(sorted_id) - sorted_id

    for i, cluster_data in enumerate(cluster_data_list):
        side = 'l' if LEFT else 'r'
        filename = f'{SEGMENT_ID}_s{SIMPLIFY}_{side}{sorted_id[i]}.obj'
        save_obj(filename, cluster_data)

def simplify_image(input_image, output_image, factor):
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(input_image)

    width, height = img.size
    new_size = (width // factor, height // factor)
    resized_img = img.resize(new_size)
    resized_img.save(output_image)
    img.close()

def re_index_merge(data):
    data['faces'] -= 1
    selected_vertices = np.unique(data['faces'][:,:,0])

    # only leave the vertices used to form the faces
    data['vertices'] = data['vertices'][selected_vertices]
    data['normals']  = data['normals'][selected_vertices]
    data['uvs']      = data['uvs'][selected_vertices]
    data['colors']   = data['colors'][selected_vertices]

    # update face index
    vertex_mapping = { old_index: new_index for new_index, old_index in enumerate(selected_vertices) }
    data['faces'] = np.vectorize(lambda x: vertex_mapping.get(x, x))(data['faces'])
    data['faces'] += 1

def merge_obj(input_list, output_filename):
    filename = input_list[0]
    merge_data = parse_obj(filename)

    for filename in input_list[1:]:
        data = parse_obj(filename)
        data['faces'] += merge_data['vertices'].shape[0]

        merge_data['vertices'] = np.vstack((merge_data['vertices'], data['vertices']))
        merge_data['normals'] = np.vstack((merge_data['normals'], data['normals']))
        merge_data['uvs'] = np.vstack((merge_data['uvs'], data['uvs']))
        merge_data['colors'] = np.vstack((merge_data['colors'], data['colors']))
        merge_data['faces'] = np.vstack((merge_data['faces'], data['faces']))

    save_obj(output_filename, merge_data)

SIMPLIFY = 100
REVERSE = False
SEGMENT_ID = '20230702185753'
CENTER_INPUT = 'scroll1_center.obj'

OBJ_INPUT = f'{SEGMENT_ID}.obj'
OBJ_SIMPLIFY = f'{SEGMENT_ID}_s{SIMPLIFY}.obj'
OBJ_SIMPLIFY_LEFT = f'{SEGMENT_ID}_s{SIMPLIFY}_l.obj'
OBJ_SIMPLIFY_RIGHT = f'{SEGMENT_ID}_s{SIMPLIFY}_r.obj'

# simplify mesh
print("simplify ...")
simplify_obj(OBJ_INPUT, OBJ_SIMPLIFY, SIMPLIFY)

# cut into half
print("cutting ...")
X_CENTER = parse_obj(CENTER_INPUT)['vertices']
cut_obj(OBJ_SIMPLIFY, X_CENTER, OBJ_SIMPLIFY_LEFT, OBJ_SIMPLIFY_RIGHT)

# divide into multiple pieces
print("clustering ...")
cluster_obj(OBJ_SIMPLIFY_LEFT, SEGMENT_ID, SIMPLIFY, REVERSE, True)
cluster_obj(OBJ_SIMPLIFY_RIGHT, SEGMENT_ID, SIMPLIFY, REVERSE, False)

# simplify image
# simplify_image('20230702185753.tif', '20230702185753_s100.tif', 10)

# merge .obj files
# input_list = ['20230702185753_s100_l1.obj', '20230702185753_s100_l4.obj']
# merge_obj(input_list, 'merge.obj')



