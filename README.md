# ABCNet: An attention-based method for particle tagging.

This is the main repository for the   [[ABCNet paper]](https://arxiv.org/abs/2001.05311).
The implementation uses a modified version of [[GAPNet]](https://arxiv.org/abs/1905.08705) to suit the High Energy Physics needs.
The implementation is divided in two main folders: classification and segmentation for the quark-gluon tagging and pileup mitigation applications, respectively. 
The input ```.h5``` files for are expected to have the following structure:

* **data**: [N,P,F], 
* **label**:[N,P]
* **pid**: [N]
* **global**: [N,G]

N = Number of events
P = Number of points
G = Number of global features

For **classification**, only the **pid** is required, while for segmentation only **label** is required.

# Classification




# License

MIT License

# Acknowledgements
ABCNet uses a modified version of [[GAPNet]](https://arxiv.org/abs/1905.08705) and [PointNet](https://github.com/charlesq34/pointnet).