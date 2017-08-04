import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="provide a base folder which will have this structure: ./01, ./01/*.csv, ./02, ./02/*.csv, ...")
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder")

    fig = plt.figure()
    pos_plot = fig.add_subplot(111, projection='3d')
    fig = plt.figure()
    ori_x = fig.add_subplot(411)
    ori_y = fig.add_subplot(412)
    ori_z = fig.add_subplot(413)
    ori_w = fig.add_subplot(414)
    fig = plt.figure()
    fx = fig.add_subplot(321)
    fy = fig.add_subplot(322)
    fz = fig.add_subplot(323)
    mx = fig.add_subplot(324)
    my = fig.add_subplot(325)
    mz = fig.add_subplot(326)


    files = os.listdir(options.base_folder)

    from matplotlib.pyplot import cm 
    import numpy as np
    color=iter(cm.rainbow(np.linspace(0, 1, len(files))))

    for f in files:
        path = os.path.join(options.base_folder, f)
        if not os.path.isdir(path):
            continue

        c = color.next()

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            csv_file_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            csv_file_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have csv file."%(path,))

        legend_name = f
        df = pd.read_csv(csv_file_path, sep=',')
        
        df = df.loc[df['.tag'] != 0]
        df.index = np.arange(1, len(df)+1)

        df['time']= pd.to_datetime(df['time'], coerce=True)
        start_time = df.head(1)['time']
        df['time']= df['time']-start_time

        pos_plot.plot(
            df['.endpoint_state.pose.position.x'].tolist(), 
            df['.endpoint_state.pose.position.y'].tolist(), 
            df['.endpoint_state.pose.position.z'].tolist(), 
            color=c,
            label=f)
        pos_plot.legend()
        pos_plot.set_title("pos xyz")

        ori_x.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.x'].tolist(), 
            color=c,
            label=f
        )
        ori_x.set_title("ori x")

        ori_y.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.y'].tolist(), 
            color=c,
            label=f
        )
        ori_y.set_title("ori y")

        ori_z.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.z'].tolist(), 
            color=c,
            label=f
        )
        ori_z.set_title("ori z")

        ori_w.plot(
            df['.endpoint_state.pose.orientation.w'].tolist(), 
            color=c,
            label=f
        )
        ori_w.legend()
        ori_w.set_title("ori w")


        fx.plot(
            df['.wrench_stamped.wrench.force.x'].tolist(), 
            color=c,
            label=f)
        fx.legend()
        fx.set_title("fx")

        fy.plot(
            df['.wrench_stamped.wrench.force.y'].tolist(), 
            color=c,
            label=f)
        fy.set_title("fy")

        fz.plot(
            df['.wrench_stamped.wrench.force.z'].tolist(), 
            color=c,
            label=f)
        fz.set_title("fz")

        mx.plot(
            df['.wrench_stamped.wrench.torque.x'].tolist(), 
            color=c,
            label=f)
        mx.set_title("mx")

        my.plot(
            df['.wrench_stamped.wrench.torque.y'].tolist(), 
            color=c,
            label=f)
        my.set_title("my")

        mz.plot(
            df['.wrench_stamped.wrench.torque.z'].tolist(), 
            color=c,
            label=f)
        mz.set_title("mz")

    plt.show()

