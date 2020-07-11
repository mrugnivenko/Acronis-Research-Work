import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt


def get_plot_fig(df, columns, list_of_anomaly, dc, inst_num, reloads):
    if 'target_der' not in df.columns.values:
        print('hmmmm it seems you are to get new ../datasets/for_plot/...')
        print('the suitable version established on 03.03.2020')
    colors = ['purple', 'pink', 'green', 'gold', 'yellow', 'blue', 'black']
    fig = make_subplots(rows=5, cols=1)
    number_of_picture_in_subplot = 3
    for column, color in zip(columns, colors):
        spec_group = False

        if column in ['target_der_sc', 'predictions_sc']:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[column],
                line_color=color,
                name=column,
                opacity=0.8), row=1, col=1)
            for rl in reloads:
                fig.add_trace(go.Scatter(
                    x=[datetime.fromtimestamp(rl), datetime.fromtimestamp(rl)],
                    y=[0, max(df['target_der_sc'])],
                    line_color='black',
                    name='reload',
                    opacity=0.8), row=1, col=1)
            spec_group = True

        if column in ['target', 'predicted_target']:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[column],
                line_color=color,
                name=column,
                opacity=0.8), row=2, col=1)
            spec_group = True

        if not spec_group:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[column],
                line_color=color,
                name=column,
                opacity=0.8), row=number_of_picture_in_subplot, col=1)
            number_of_picture_in_subplot += 1
    plot_anomaly(df, list_of_anomaly, fig)
    fig.update_layout(height=1200, width=800, title_text='dc = ' + dc + ' inst_num = ' + str(inst_num))
    # plotly.offline.plot(fig,filename='sampleplot.html')
    return fig


def plot_anomaly(df, list_of_anomaly, fig):
    i = 0
    for anomaly in list_of_anomaly:
        if i == 0:
            fig.add_trace(go.Scatter(
                x=df['date'].loc[
                  list(df[df['time_stamp'] == anomaly[0]].index)[0]:list(df[df['time_stamp'] == anomaly[1]].index)[0]],
                y=df['target_der_sc'].loc[
                  list(df[df['time_stamp'] == anomaly[0]].index)[0]:list(df[df['time_stamp'] == anomaly[1]].index)[0]],
                line_color='red',
                line_width=3,
                name='anomaly',
                opacity=0.9), row=1, col=1)
            i += 1
        else:
            fig.add_trace(go.Scatter(
                x=df['date'].loc[
                  list(df[df['time_stamp'] == anomaly[0]].index)[0]:list(df[df['time_stamp'] == anomaly[1]].index)[0]],
                y=df['target_der_sc'].loc[
                  list(df[df['time_stamp'] == anomaly[0]].index)[0]:list(df[df['time_stamp'] == anomaly[1]].index)[0]],
                line_color='red',
                line_width=3,
                showlegend=False,
                opacity=0.9), row=1, col=1)


def get_ez_ram_plt(data, list_of_anomalys, reloads=[], reload_window=600):
    fig, axs = plt.subplots(5, 1, figsize=(12, 24))
    axs[0].plot(data['date'], data['predictions_sc'], label='predicted_target_der_scaled')
    axs[0].plot(data['date'], data['target_der_sc'], label='target_der_scaled')
    axs[0].set_xlabel('date')
    axs[0].set_ylabel('derivate of target')
    axs[0].grid(True)

    a_f = 0
    for an in list_of_anomalys:
        a_f += 1
        add_d = data[data['time_stamp'] >= an[0]]
        add_d = add_d[add_d['time_stamp'] <= an[1]]
        if a_f == 1:
            axs[0].plot(add_d['date'], add_d['predictions_sc'], color='red', label='anomaly')
        else:
            axs[0].plot(add_d['date'], add_d['predictions_sc'], color='red')
    axs[0].legend(fontsize=10)

    a_f = 0
    for rel in reloads:
        a_f += 1
        add_d = data[data['time_stamp'] >= rel + reload_window]
        add_d = add_d[add_d['time_stamp'] <= rel - reload_window]
        if a_f == 1:
            axs[0].plot(add_d['date'], add_d['predictions_sc'], color='green', label='reload')
        else:
            axs[0].plot(add_d['date'], add_d['predictions_sc'], color='green')
        axs[0].axvline(x=datetime.fromtimestamp(rel), color='green')
    axs[0].legend(fontsize=10)

    axs[1].plot(data['date'], data['target'], label='target')
    axs[1].plot(data['date'], data['predicted_target'], label='predicted_target')
    axs[1].set_xlabel('date')
    axs[1].set_ylabel('target')
    axs[1].grid(True)
    axs[1].legend(fontsize=10)

    axs[2].plot(data['date'], data['main_error'], label='main_error')
    axs[2].set_xlabel('date')
    axs[2].set_ylabel('main_error')
    axs[2].grid(True)
    axs[2].legend(fontsize=10)

    axs[3].plot(data['date'], data['variance'], label='variance')
    axs[3].set_xlabel('date')
    axs[3].set_ylabel('variance')
    axs[3].grid(True)
    axs[3].legend(fontsize=10)

    axs[4].plot(data['date'], data['mean'], label='mean_latency')
    axs[4].set_xlabel('date')
    axs[4].set_ylabel('mean_latency')
    axs[4].grid(True)
    axs[4].legend(fontsize=10)

    return fig, axs
