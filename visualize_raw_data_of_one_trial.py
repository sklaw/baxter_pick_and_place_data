import os
import pandas as pd
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 
import numpy as np
import load_data_folder
import plot_data_in_panda_df


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

    df_group_by_foldername = load_data_folder.run(options.base_folder)
    
    f, df = df_group_by_foldername.iteritems().next()
    
    df = df.loc[df['.tag'] != 0]
    df.index = np.arange(1, len(df)+1)

    df['time']= pd.to_datetime(df['time'], coerce=True)
    start_time = df.head(1)['time']
    df['time']= df['time']-start_time

    state_amount = len(df['.tag'].unique())
    color=iter(cm.rainbow(np.linspace(0, 1, state_amount)))

    plot_data_in_panda_df.init_plots()
    for state_no in df['.tag'].unique():
        c=next(color)
        state_df = df.loc[df['.tag'] == state_no]
        plot_data_in_panda_df.plot_one_df(state_df, color=c, label=f)
    plot_data_in_panda_df.show_plots()

