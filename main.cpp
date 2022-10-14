#include <unordered_set>
#include <filesystem>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "happly.h"
#include "bvh/triangle.hpp"
#include "bvh/sweep_sah_builder.hpp"
#include "bvh/single_ray_traverser.hpp"
#include "bvh/primitive_intersectors.hpp"

using Vector3  = bvh::Vector3<float>;
using Triangle = bvh::Triangle<float>;
using Ray = bvh::Ray<float>;
using Bvh = bvh::Bvh<float>;

constexpr int width = 10;
constexpr int height = 10;

std::vector<Triangle> parse_obj(const std::string &filepath) {
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return {};
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    std::vector<Triangle> triangles;

    // Loop over shapes
    for (const auto &shape : shapes) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            assert(shape.mesh.num_face_vertices[f] == 3);

            // Loop over vertices in the face.
            tinyobj::index_t idx_1 = shape.mesh.indices[index_offset];
            float vx_1 = attrib.vertices[3 * size_t(idx_1.vertex_index) + 0];
            float vy_1 = attrib.vertices[3 * size_t(idx_1.vertex_index) + 1];
            float vz_1 = attrib.vertices[3 * size_t(idx_1.vertex_index) + 2];
            index_offset += 1;

            tinyobj::index_t idx_2 = shape.mesh.indices[index_offset];
            float vx_2 = attrib.vertices[3 * size_t(idx_2.vertex_index) + 0];
            float vy_2 = attrib.vertices[3 * size_t(idx_2.vertex_index) + 1];
            float vz_2 = attrib.vertices[3 * size_t(idx_2.vertex_index) + 2];
            index_offset += 1;

            tinyobj::index_t idx_3 = shape.mesh.indices[index_offset];
            float vx_3 = attrib.vertices[3 * size_t(idx_3.vertex_index) + 0];
            float vy_3 = attrib.vertices[3 * size_t(idx_3.vertex_index) + 1];
            float vz_3 = attrib.vertices[3 * size_t(idx_3.vertex_index) + 2];
            index_offset += 1;

            triangles.emplace_back(Vector3(vx_1, vy_1, vz_1),
                                   Vector3(vx_2, vy_2, vz_2),
                                   Vector3(vx_3, vy_3, vz_3));
        }
    }

    return triangles;
}

std::vector<Triangle> parse_ply(const std::string &filepath) {
    happly::PLYData ply_data(filepath);
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<Triangle> triangles;
    for (auto &face : f_idx) {
        triangles.emplace_back(Vector3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                               Vector3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                               Vector3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]));
    }
    return triangles;
}

int main() {
    std::vector<Triangle> triangles = parse_obj("../sponza.obj");

    Bvh bvh;
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    std::cout << "global bounding box: ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")";

    bvh::SweepSahBuilder<Bvh> builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::vector<std::pair<size_t, size_t>> edges;
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        queue.pop();
        if (!bvh.nodes[curr].is_leaf()) {
            size_t left_idx = bvh.nodes[curr].first_child_or_primitive;
            size_t right_idx = left_idx + 1;
            edges.emplace_back(curr, left_idx);
            edges.emplace_back(curr, right_idx);
            queue.push(left_idx);
            queue.push(right_idx);
        }
    }

    bvh::ClosestPrimitiveIntersector<Bvh, Triangle> primitive_intersector(bvh, triangles.data());
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    std::ofstream image_file("image.ppm");
    image_file << "P3\n" << width << ' ' << height << "\n255\n";
    std::filesystem::create_directory("graphs");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            constexpr float origin_x = 5.f;
            constexpr float origin_y = 7.f;
            constexpr float origin_z = 0.f;
            float dir_x = -1.f;
            float dir_y = (8.f - 2.f * i / height) - origin_y;
            float dir_z = (1.f - 2.f * j / width) - origin_z;
            Ray ray(
                    Vector3(origin_x, origin_y, origin_z),
                    Vector3(dir_x, dir_y, dir_z),
                    0.f
            );

            std::unordered_set<size_t> traversed;
            if (auto hit = traverser.traverse(ray, primitive_intersector, traversed)) {
                auto triangle_index = hit->primitive_index;
                float r = triangles[triangle_index].n[0];
                float g = triangles[triangle_index].n[1];
                float b = triangles[triangle_index].n[2];
                float length = sqrtf(r * r + g * g + b * b);
                r = (r / length + 1.f) / 2.f;
                g = (g / length + 1.f) / 2.f;
                b = (b / length + 1.f) / 2.f;
                image_file << std::clamp(int(256.f * r), 0, 255) << ' '
                           << std::clamp(int(256.f * g), 0, 255) << ' '
                           << std::clamp(int(256.f * b), 0, 255) << '\n';
            } else {
                image_file << "0 0 0\n";
            }

            std::string filepath = "graphs/bvh_" + std::to_string(i) + "_" + std::to_string(j) + ".gv";
            std::ofstream gv_file(filepath);
            gv_file << "digraph bvh {\n";
            gv_file << "    layout=twopi\n";
            gv_file << "    ranksep=2\n";
            gv_file << "    node [shape=point]\n";
            gv_file << "    edge [arrowhead=none penwidth=0.5]\n";
            gv_file << "    0 [shape=circle label=root]";

            for (auto &edge : edges) {
                gv_file << "\n    " << edge.first << " -> " << edge.second;
                if (traversed.count(edge.first) != 0) gv_file << " [color=red penwidth=5]";
            }

            gv_file << "\n}";
            gv_file.close();
        }
    }
}
