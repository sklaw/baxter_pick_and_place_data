import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 
import numpy as np



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
    ori_xyz = fig.add_subplot(211, projection='3d')

    ori_x = fig.add_subplot(411)
    ori_y = fig.add_subplot(412)
    ori_z = fig.add_subplot(413)
    ori_w = fig.add_subplot(414)

    fig = plt.figure()
    fx = fig.add_subplot(611)
    fy = fig.add_subplot(612)
    fz = fig.add_subplot(613)
    mx = fig.add_subplot(614)
    my = fig.add_subplot(615)
    mz = fig.add_subplot(616)


    files = os.listdir(options.base_folder)
    for f in files:
        path = os.path.join(options.base_folder, f)
        if not os.path.isdir(path):
            continue
        if f.startswith("bad"):
            continue
    

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            csv_file_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            csv_file_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have csv file."%(path,))
        break

    legend_name = f
    df = pd.read_csv(csv_file_path, sep=',')
    
    df = df.loc[df['.tag'] != 0]
    df.index = np.arange(1, len(df)+1)

    df['time']= pd.to_datetime(df['time'], coerce=True)
    start_time = df.head(1)['time']
    df['time']= df['time']-start_time

    state_amount = len(df['.tag'].unique())


    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))
    for state_no in df['.tag'].unique():
        c=next(color)

        state_df = df.loc[df['.tag'] == state_no]

        pos_plot.plot(
            state_df['.endpoint_state.pose.position.x'].tolist(), 
            state_df['.endpoint_state.pose.position.y'].tolist(), 
            state_df['.endpoint_state.pose.position.z'].tolist(), 
            color=c,
            label=f
        )
        pos_plot.set_title("pos xyz")

        ori_x.plot(
            state_df.index.tolist(),
            state_df['.endpoint_state.pose.orientation.x'].tolist(), 
            color=c,
            label=f
        )
        ori_x.set_title("ori x")

        ori_y.plot(
            state_df.index.tolist(),
            state_df['.endpoint_state.pose.orientation.y'].tolist(), 
            color=c,
            label=f
        )
        ori_y.set_title("ori y")

        ori_z.plot(
            state_df.index.tolist(),
            state_df['.endpoint_state.pose.orientation.z'].tolist(), 
            color=c,
            label=f
        )
        ori_z.set_title("ori z")

        ori_w.plot(
            state_df.index.tolist(),
            state_df['.endpoint_state.pose.orientation.w'].tolist(), 
            color=c,
            label=f
        )
        ori_w.set_title("ori w")


        fx.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.force.x'].tolist(), 
            color=c,
            label=f)
        fx.set_title("fx")

        fy.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.force.y'].tolist(), 
            color=c,
            label=f)
        fy.set_title("fy")

        fz.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.force.z'].tolist(), 
            color=c,
            label=f)
        fz.set_title("fz")

        mx.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.torque.x'].tolist(), 
            color=c,
            label=f)
        mx.set_title("mx")

        my.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.torque.y'].tolist(), 
            color=c,
            label=f)
        my.set_title("my")

        mz.plot(
            state_df.index.tolist(),
            state_df['.wrench_stamped.wrench.torque.z'].tolist(), 
            color=c,
            label=f)
        mz.set_title("mz")

    plt.show()

