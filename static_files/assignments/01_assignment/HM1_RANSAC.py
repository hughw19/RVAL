import numpy as np
from utils import draw_save_plane_with_points


if __name__ == "__main__":


    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    

    sample_time = #more than 99.9% probability at least one hypothesis does not contain any outliers 
    distance_threshold = 0.05

    # sample points group




    # estimate the plane with sampled points group




    #evaluate inliers (with point-to-plance distance < distance_threshold)



    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 



    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

