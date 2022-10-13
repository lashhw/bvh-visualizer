#include <map>
#include <unordered_set>
#include <filesystem>
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
constexpr float origin_x = 0.f;
constexpr float origin_y = 0.1f;
constexpr float origin_z = 1.f;
constexpr float horizontal = 0.2f;
constexpr float vertical = 0.2f;

int main() {
    happly::PLYData ply_data("../bun_zipper_res3.ply");
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<Triangle> triangles;
    for (auto &face : f_idx) {
        triangles.emplace_back(Vector3(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                               Vector3(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                               Vector3(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]));
    }

    Bvh bvh;
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());

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

    std::filesystem::create_directory("graphs");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float dir_x = (-0.1f + horizontal * j / width) - origin_x;
            float dir_y = (0.2f - vertical * i / height) - origin_y;
            float dir_z = -1.f;
            Ray ray(
                    Vector3(origin_x, origin_y, origin_z),
                    Vector3(dir_x, dir_y, dir_z),
                    0.f
            );

            std::unordered_set<size_t> traversed;
            traverser.traverse(ray, primitive_intersector, traversed);

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
