import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import imageio

iris_data_set = load_iris()                     # load the iris dataset

x = iris_data_set.data
y_raw=iris_data_set.target
y = y_raw.reshape(-1, 1)        # Convert data to a single column

lr=LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

x_train, x_test, y_train, y_test = train_test_split(
             x, y, test_size=0.2,
             random_state=13
)

lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x)
wrong_prediction_lr_indexes=np.asarray(np.where(y_pred_lr.T != y_raw))[0]

# plotting the result for Logistic Regression

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


plt.rcParams.update({'font.size': 7})
labelTups = [(iris_data_set.target_names[0], 0), (iris_data_set.target_names[1], 1), (iris_data_set.target_names[2], 2)]
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
# plot ox0y0z0 axes

for item in wrong_prediction_lr_indexes.T:
    # print(item)
    y_item=y_raw[item]
    ax.text3D(x[:, 0].mean() + y_item,
              x[:, 1].mean() + 1.5 - y_item,
              x[:, 2].mean(), 'Ground Truth: '+iris_data_set.target_names[y_item],
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    a = Arrow3D([x[:, 0].mean()+ y_item, x[item, 0].mean()],
                [x[:, 1].mean()+ 1.5 - y_item , x[item, 1].mean()],
                [x[:, 2].mean(), x[item, 2].mean()], **arrow_prop_dict)
    ax.add_artist(a)

sc=ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_pred_lr,
            edgecolor='k', label=[lt[0] for lt in labelTups])
plt.title('IRIS classification via Logistic Regression. The accuracy is: '+ str(1-wrong_prediction_lr_indexes.size/y.size))
ax.set_xlabel(iris_data_set.feature_names[0])
ax.set_ylabel(iris_data_set.feature_names[1])
ax.set_zlabel(iris_data_set.feature_names[2])
colors = [sc.cmap(sc.norm(i)) for i in [0, 1, 2]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],
          bbox_to_anchor=(0.3, .75))
fig.tight_layout()
plt.show()

#animating the result
def plot_for_offset(angle):
    # Data for plotting
    ax.view_init(30, angle)
    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

# kwargs_write = {'fps':10, 'quantizer':'nq'}
imageio.mimsave('./IRIS_LR.gif', [plot_for_offset(i) for i in range(0,360,3)], fps=10)