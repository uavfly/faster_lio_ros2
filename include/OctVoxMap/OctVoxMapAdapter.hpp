#ifndef OCTVOXMAP_ADAPTER_HPP
#define OCTVOXMAP_ADAPTER_HPP

#include "OctVoxMap.hpp"
#include <vector>
#include <memory>
#include <pcl/point_types.h>
#include <Eigen/Core>

// Adapter to make OctVoxMap look like IVox
template <int dim = 3, int node_type = 0, typename PointType = pcl::PointXYZ>
class OctVoxMapAdapter {
public:
    using KeyType = Eigen::Matrix<int, dim, 1>;
    using PtType = Eigen::Matrix<float, dim, 1>;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    
    // Use Eigen::Vector3f internally for OctVoxMap to avoid operator overloading issues with PCL types
    using InternalPointType = Eigen::Matrix<float, dim, 1>;
    using OctVoxMapType = LI2Sup::OctVoxMap<InternalPointType, float>;
    using KNNHeapType = typename OctVoxMapType::KNNHeapType;

    enum class NearbyType {
        CENTER,
        NEARBY6,
        NEARBY18,
        NEARBY26
    };

    struct Options {
        float resolution_ = 0.2;
        float inv_resolution_ = 10.0;
        NearbyType nearby_type_ = NearbyType::NEARBY6; // Not used by OctVoxMap
        std::size_t capacity_ = 1000000;
    };

    explicit OctVoxMapAdapter(Options options) {
        typename OctVoxMapType::Options oct_options(options.resolution_, options.capacity_);
        oct_vox_map_.reset(new OctVoxMapType(oct_options));
    }

    void AddPoints(const PointVector& points_to_add) {
        // Convert PCL points to Eigen points
        std::vector<InternalPointType, Eigen::aligned_allocator<InternalPointType>> internal_points;
        internal_points.reserve(points_to_add.size());
        for (const auto& pt : points_to_add) {
            internal_points.emplace_back(pt.x, pt.y, pt.z);
        }
        
        // OctVoxMap handles downsampling internally
        oct_vox_map_->insert(internal_points);
    }

    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0) {
        // OctVoxMap hardcodes K=5 in KNNHeap<5, Point>
        if (max_num > 5) max_num = 5; 

        InternalPointType internal_pt(pt.x, pt.y, pt.z);
        KNNHeapType top_K;
        
        // OctVoxMap::getTopK takes (Point, KNNHeap&)
        oct_vox_map_->getTopK(internal_pt, top_K);

        closest_pt.clear();
        closest_pt.reserve(top_K.count);

        float max_range_sq = max_range * max_range;
        
        for (int i = 0; i < top_K.count; ++i) {
            if (top_K.dist2_[i] <= max_range_sq) {
                PointType p;
                p.x = top_K.points_[i].x();
                p.y = top_K.points_[i].y();
                p.z = top_K.points_[i].z();
                // Intensity/Curvature are lost, but not needed for geometric ICP
                closest_pt.emplace_back(p);
            }
        }
        
        return !closest_pt.empty();
    }
    
    // Overload for single point NN if needed
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt) {
        InternalPointType internal_pt(pt.x, pt.y, pt.z);
        KNNHeapType top_K;
        oct_vox_map_->getTopK(internal_pt, top_K);
        if (top_K.count > 0) {
             // Find the closest one in the heap
             int best_idx = -1;
             float min_dist = std::numeric_limits<float>::max();
             for(int i=0; i<top_K.count; ++i) {
                 if(top_K.dist2_[i] < min_dist) {
                     min_dist = top_K.dist2_[i];
                     best_idx = i;
                 }
             }
             if (best_idx != -1) {
                 closest_pt.x = top_K.points_[best_idx].x();
                 closest_pt.y = top_K.points_[best_idx].y();
                 closest_pt.z = top_K.points_[best_idx].z();
                 return true;
             }
        }
        return false;
    }

    size_t NumValidGrids() const { return 0; } 
    size_t NumPoints() const { return 0; }
    std::vector<float> StatGridPoints() const { return {}; }

private:
    std::shared_ptr<OctVoxMapType> oct_vox_map_;
};

#endif // OCTVOXMAP_ADAPTER_HPP
