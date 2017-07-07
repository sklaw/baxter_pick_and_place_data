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
    ori_plot = fig.add_subplot(111, projection='3d')

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

        legend_name = f
        df = pd.read_csv(csv_file_path, sep=',')
        
        df = df[[
            u'time', 
            u'.endpoint_state.pose.position.x',
            u'.endpoint_state.pose.position.y',
            u'.endpoint_state.pose.position.z',
            u'.endpoint_state.pose.orientation.x',
            u'.endpoint_state.pose.orientation.y',
            u'.endpoint_state.pose.orientation.z',
            u'.endpoint_state.pose.orientation.w',
            u'.tag']]

        df = df.loc[df['.tag'] != 0]
        df['time']= pd.to_datetime(df['time'], coerce=True)
        start_time = df.head(1)['time']
        df['time']= df['time']-start_time

        pos_plot.plot(
            df['.endpoint_state.pose.position.x'].tolist(), 
            df['.endpoint_state.pose.position.y'].tolist(), 
            df['.endpoint_state.pose.position.z'].tolist(), 
            label=f)
        pos_plot.legend()
        pos_plot.set_title("pos xyz")

        ori_plot.plot(
            df['.endpoint_state.pose.orientation.x'].tolist(), 
            df['.endpoint_state.pose.orientation.y'].tolist(), 
            df['.endpoint_state.pose.orientation.z'].tolist(), 
            label=f)
        ori_plot.legend()
        ori_plot.set_title("ori xyz")
    plt.show()

