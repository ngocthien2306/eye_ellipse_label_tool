# Kỹ thuật loại bỏ điểm nhiễu trước khi fit ellipse
import numpy as np


def clean_contour_points(contour_points_cv, method='ransac'):
    if len(contour_points_cv) < 5:
        return contour_points_cv
    
    if method == 'ransac':
        # Sử dụng RANSAC để loại bỏ outliers
        from skimage.measure import ransac, EllipseModel
        
        model = EllipseModel()
        points = contour_points_cv.astype(np.float64)
        
        # Thực hiện RANSAC
        model_robust, inliers = ransac(points, EllipseModel, min_samples=5,
                                      residual_threshold=2.0, max_trials=100)
        
        # Trả về chỉ các điểm inliers
        if model_robust is not None and inliers is not None:
            return points[inliers]
        return contour_points_cv
    
    elif method == 'distance_filter':
        # Lọc theo khoảng cách trung bình
        from scipy.spatial.distance import pdist, squareform
        
        points = contour_points_cv.astype(np.float64)
        
        # Tính ma trận khoảng cách giữa các điểm
        distances = squareform(pdist(points))
        
        # Tính khoảng cách trung bình của mỗi điểm đến các điểm khác
        mean_distances = np.mean(distances, axis=1)
        
        # Tính ngưỡng (có thể điều chỉnh hệ số 1.5)
        threshold = np.mean(mean_distances) + 1.5 * np.std(mean_distances)
        
        # Giữ lại các điểm có khoảng cách trung bình dưới ngưỡng
        filtered_indices = np.where(mean_distances < threshold)[0]
        return points[filtered_indices]
    
    elif method == 'statistical':
        # Lọc theo độ lệch chuẩn
        points = contour_points_cv.astype(np.float64)
        
        # Tính tâm của điểm
        center = np.mean(points, axis=0)
        
        # Tính khoảng cách từ mỗi điểm đến tâm
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        
        # Tính ngưỡng (có thể điều chỉnh hệ số 2.0)
        threshold = np.mean(distances) + 2.0 * np.std(distances)
        
        # Giữ lại các điểm có khoảng cách đến tâm dưới ngưỡng
        filtered_indices = np.where(distances < threshold)[0]
        return points[filtered_indices]
    
    elif method == 'convex_hull':
        # Sử dụng convex hull để loại bỏ các điểm không thuộc hull
        from scipy.spatial import ConvexHull
        
        points = contour_points_cv.astype(np.float64)
        
        if len(points) >= 3:  # Convex hull cần ít nhất 3 điểm
            hull = ConvexHull(points)
            return points[hull.vertices]
        
        return points
    
    return contour_points_cv