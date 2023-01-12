import numpy as np

import logging
logging.getLogger('PIL').setLevel(logging.CRITICAL)
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import gif

class MPEAnimator():
    
    def __init__(self,
                 agent_positions, 
                 landmark_positions,
                 episode_rewards,
                 mask_agents=False):
        
        self.n_agents = len(agent_positions)
        self.n_landmarks = len(landmark_positions)
        self.n_frames = agent_positions.shape[1]
        
        self.agent_positions = agent_positions
        self.landmark_positions = landmark_positions
        self.episode_rewards = episode_rewards
        self.mask_agents = mask_agents
        self.xy_max = 0
        
        
    def animate(self):
        self.frames = []
        for f in range(self.n_frames):
            if self.mask_agents:
                self.frame = self.plot_static_noagents(f)
            else:
                self.frame = self.plot_static(f)
            self.frames.append(self.frame)
        return self.frames

    
    def save_animation(self, savepath='episode'):
        _ = self.animate()
        self.frame.save(savepath+'.png')
        gif.save(self.frames, savepath+'.gif', duration=50, unit="ms", between="frames")
            
    @gif.frame
    def plot_static(self, frame):
        
        """
        Create the plolty figure
        """
        
        fig = make_subplots(rows=2, cols=2,
                            specs=[[{"rowspan": 2}, {}],
                                   [None, {}]],
                            subplot_titles=("Episode", "Reward", "Plot space for sale"))

        for a in range(self.n_agents):
            # agent head
            fig.add_trace(go.Scatter(x=self.agent_positions[a, frame-1:frame, 0],
                                     y=self.agent_positions[a, frame-1:frame, 1],
                                     mode='markers',
                                     marker=dict(size=10,
                                                 color=plotly.colors.DEFAULT_PLOTLY_COLORS[a],
                                                 line=dict(width=1)),
                                     name=f'Agent {a+1}'),
                         row=1, col=1)

            # agent trajectory
            fig.add_trace(go.Scatter(x=self.agent_positions[a, :frame, 0],
                                     y=self.agent_positions[a, :frame, 1],
                                     mode='markers',
                                     showlegend=False,
                                     marker=dict(size=5, color=plotly.colors.DEFAULT_PLOTLY_COLORS[a], opacity=0.2)), 
                         row=1, col=1)


        # landmarks
        for l in range(self.n_landmarks):  
            # landmark real head
            fig.add_trace(go.Scatter(x=self.landmark_positions[l, frame-1:frame, 0],
                                     y=self.landmark_positions[l, frame-1:frame, 1],
                                     mode='markers',
                                     marker_symbol='diamond',
                                     marker=dict(size=10, color=px.colors.sequential.Greens[::-1][l]),
                                     name=f'Landmark {l+1} real'),
                         row=1, col=1)

            fig.add_trace(go.Scatter(x=self.landmark_positions[l, :frame, 0],
                                     y=self.landmark_positions[l, :frame, 1],
                                     mode='markers',
                                     marker_symbol='diamond',
                                     showlegend=False,
                                     marker=dict(size=10, color=px.colors.sequential.Greens[::-1][l], opacity=0.1)),
                         row=1, col=1)

        # reward
        fig.add_trace(go.Scatter(x=np.arange(frame),
                                 y=self.episode_rewards[:frame],
                                 mode='lines',
                                 marker=dict(color="green"),
                                 name='Reward'),
                     row=1, col=2)


        self._update_range(frame)
        fig.update_layout(xaxis1=dict(title='X position',range=[self.x_min-0.5, self.x_max+0.5]),
                          yaxis1=dict(title='Y position',range=[self.y_min-0.5, self.y_max+0.5]),
                          xaxis2=dict(title='Timestep'),
                          xaxis3=dict(title='Timestep'))

        fig.update_layout(autosize=False, width=1200, height=600, plot_bgcolor='white')
        return fig 


    @gif.frame
    def plot_static_noagents(self, frame):
        
        """
        As above but single plot with landmark movements
        """

        layout = go.Layout(
            margin=go.layout.Margin(
                    l=2, #left margin
                    r=2, #right margin
                    b=2, #bottom margin
                    t=2  #top margin
                )
            )
        fig = go.Figure(layout=layout)

        # landmarks
        for l in range(self.n_landmarks):  
            # landmark real head
            fig.add_trace(go.Scatter(x=self.landmark_positions[l, frame-1:frame, 0],
                                     y=self.landmark_positions[l, frame-1:frame, 1],
                                     mode='markers',
                                     marker_symbol='diamond',
                                     marker=dict(size=10, color=px.colors.sequential.Greens[::-1][l]),
                                     name=f'Landmark {l+1} real'))

            fig.add_trace(go.Scatter(x=self.landmark_positions[l, :frame, 0],
                                     y=self.landmark_positions[l, :frame, 1],
                                     mode='markers',
                                     marker_symbol='diamond',
                                     showlegend=False,
                                     marker=dict(size=10, color=px.colors.sequential.Greens[::-1][l], opacity=0.1)))

        fig.update_layout(xaxis1=dict(title='X position'),
                          yaxis1=dict(title='Y position'))

        fig.update_layout(autosize=False, width=600, height=600, plot_bgcolor='white')
        return fig 
        
          
    def _update_range(self, frame):
        
        frame = max(1, frame)
        self.x_max = max(self.agent_positions[:,:frame,0].max(),
                         self.landmark_positions[:,:frame,0].max())

        self.x_min = min(self.agent_positions[:,:frame,0].min(),
                         self.landmark_positions[:,:frame,0].min())

        self.y_max = max(self.agent_positions[:,:frame,1].max(),
                         self.landmark_positions[:,:frame,1].max())

        self.y_min = min(self.agent_positions[:,:frame,1].min(),
                         self.landmark_positions[:,:frame,1].min())

        self.xy_max = max(self.xy_max, self.x_max, self.y_max)
        self.x_max = self.xy_max
        self.y_max = self.xy_max