import numpy as np
import open3d as o3d

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

def simplify_obj(INPUT_OBJ, OUTPUT_OBJ, SIMPLIFY):
    data = parse_obj(INPUT_OBJ)

    # Open3D decimation can't deal with UV info, so let's save those info as vertices colors (it's a trick :P)
    uvs_to_colors(data)
    save_obj(OUTPUT_OBJ, data)

    # mesh decimation
    mesh = o3d.io.read_triangle_mesh(OUTPUT_OBJ)
    TRI_NUM = len(mesh.triangles) // SIMPLIFY
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=TRI_NUM)

    # Once Open3D finishes decimation, restore those uv info from vertices colors
    data = mesh_to_data(mesh_smp)
    colors_to_uvs(data)
    save_obj(OUTPUT_OBJ, data)

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

# SIMPLIFY = 10
# SEGMENT_ID = '20230510153006'

# INPUT_OBJ = f'{SEGMENT_ID}.obj'
# OUTPUT_OBJ = f'{SEGMENT_ID}_s{SIMPLIFY}.obj'

# simplify_obj(INPUT_OBJ, OUTPUT_OBJ, SIMPLIFY)


# x_crop = 2779
# filename = '20230503225234.obj'
# x_crop = 3884
# filename = '20230510153006_s10.obj'

# data = parse_obj(filename)

# mask = np.sum(data['vertices'][data['faces'][:,:,0] - 1, 0] < x_crop, axis=1) >= 2

# data['faces'] = data['faces'][mask]

# save_obj('ok.obj', data)

data = parse_obj('ok.obj')
mesh = o3d.io.read_triangle_mesh('ok.obj')

triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
# print(np.max(triangle_clusters))

cluster_data = data.copy()
cluster_data['faces'] = data['faces'][triangle_clusters == 0]
re_index(cluster_data)
save_obj('ok-0.obj', cluster_data)

cluster_data = data.copy()
cluster_data['faces'] = data['faces'][triangle_clusters == 1]
re_index(cluster_data)
save_obj('ok-1.obj', cluster_data)

cluster_data = data.copy()
cluster_data['faces'] = data['faces'][triangle_clusters == 2]
re_index(cluster_data)
save_obj('ok-2.obj', cluster_data)

cluster_data = data.copy()
cluster_data['faces'] = data['faces'][triangle_clusters == 3]
re_index(cluster_data)
save_obj('ok-3.obj', cluster_data)




