import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ori_config import OUTPUTS_DIR  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pitch_control import PitchControl

    



class Visual:
    def __init__(self, pitch_control: 'PitchControl'):
        self.pc = pitch_control

    @staticmethod
    def convert_to_top_left_coords(x_center, y_center, field_dimen=(106.0, 68.0)):
        """
        Convert center-origin coordinates to top-left-origin pitch coordinates.

        Parameters:
        - x_center (float): x coordinate with origin at center (range: -53 to 53)
        - y_center (float): y coordinate with origin at center (range: -34 to 34)
        - field_dimen (tuple): (field length in meters, field width in meters)

        Returns:
        - (x_new, y_new): coordinates in top-left pitch system (0,0 to 105,68)
        """
        x_new = x_center + field_dimen[0] / 2
        y_new = y_center + field_dimen[1] / 2
        return x_new, y_new


    def plot_pitch(self, field_dimen = (106.0,68.0), field_color ='green', linewidth=2, markersize=20):
        """ plot_pitch
        
        Plots a soccer pitch. All distance units converted to meters.
        
        Parameters
        -----------
            field_dimen: (length, width) of field in meters. Default is (106,68)
            field_color: color of field. options are {'green','white'}
            linewidth  : width of lines. default = 2
            markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
            
        Returrns
        -----------
        fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """
        fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
        # decide what color we want the field to be. Default is green, but can also choose white
        if field_color=='green':
            ax.set_facecolor('mediumseagreen')
            lc = 'whitesmoke' # line color
            pc = 'w' # 'spot' colors
        elif field_color=='white':
            lc = 'k'
            pc = 'k'
        # ALL DIMENSIONS IN m
        border_dimen = (3,3) # include a border arround of the field of width 3m
        meters_per_yard = 0.9144 # unit conversion from yards to meters
        half_pitch_length = field_dimen[0]/2. # length of half pitch
        half_pitch_width = field_dimen[1]/2. # width of half pitch
        signs = [-1,1] 
        # Soccer field dimensions typically defined in yards, so we need to convert to meters
        goal_line_width = 8*meters_per_yard
        box_width = 20*meters_per_yard
        box_length = 6*meters_per_yard
        area_width = 44*meters_per_yard
        area_length = 18*meters_per_yard
        penalty_spot = 12*meters_per_yard
        corner_radius = 1*meters_per_yard
        D_length = 8*meters_per_yard
        D_radius = 10*meters_per_yard
        D_pos = 12*meters_per_yard
        centre_circle_radius = 10*meters_per_yard
        # plot half way line # center circle
        ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        y = np.linspace(-1,1,50)*centre_circle_radius
        x = np.sqrt(centre_circle_radius**2-y**2)
        ax.plot(x,y,lc,linewidth=linewidth)
        ax.plot(-x,y,lc,linewidth=linewidth)
        for s in signs: # plots each line seperately
            # plot pitch boundary
            ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
            ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
            # goal posts & line
            ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
            # 6 yard box
            ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
            ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
            ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
            # penalty area
            ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
            ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
            ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
            # penalty spot
            ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
            # corner flags
            y = np.linspace(0,1,50)*corner_radius
            x = np.sqrt(corner_radius**2-y**2)
            ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
            ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
            # draw the D
            y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
            x = np.sqrt(D_radius**2-y**2)+D_pos
            ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)
            
        # remove axis labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        # set axis limits
        xmax = field_dimen[0]/2. + border_dimen[0]
        ymax = field_dimen[1]/2. + border_dimen[1]
        ax.set_xlim([-xmax,xmax])
        ax.set_ylim([-ymax,ymax])
        ax.set_axisbelow(True)
        return fig,ax
    
    def plot_frame(
        self, hometeam, awayteam, figax=None,
        team_colors=('r', 'b'), field_dimen=(106.0, 68.0),
        include_player_velocities=False, PlayerMarkerSize=10,
        PlayerAlpha=0.7, annotate=False, annotate_control=True
    ):
        """
        Plots a frame of tracking data with optional player pitch control annotations.
        """

        PPCF = self.pc.PPCF_data[0]['PPCF']
        xgrid = self.pc.PPCF_data[0]['xgrid']
        ygrid = self.pc.PPCF_data[0]['ygrid']

        if figax is None:
            fig, ax = self.plot_pitch(field_dimen=field_dimen)
        else:
            fig, ax = figax


        # plot home & away teams
        for team, color in zip([hometeam, awayteam], team_colors):
            x_columns = [c for c in team.keys() if c.endswith('_x') and c != 'ball_x']
            y_columns = [c for c in team.keys() if c.endswith('_y') and c != 'ball_y']

            for x_col, y_col in zip(x_columns, y_columns):
                x = team[x_col]
                y = team[y_col]

                

                # Skip if position is NaN
                if np.isnan(x) or np.isnan(y):
                    continue

                # Plot player
                ax.plot(x, y, color + 'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)

                # Plot velocity vector
                if include_player_velocities:
                    vx_col = f'{x_col[:-2]}_vx'
                    vy_col = f'{y_col[:-2]}_vy'
                    ax.quiver(x, y, team[vx_col], team[vy_col], color=color,
                            scale_units='inches', scale=10., width=0.0015,
                            headlength=5, headwidth=3, alpha=PlayerAlpha)

                # Annotate player number
                if annotate:
                    player_number = x_col.split('_')[1]
                    ax.text(x + 0.5, y + 0.5, player_number, fontsize=10, color=color)

                # Annotate control value
                if annotate_control:
                    
                    x_new, y_new = self.convert_to_top_left_coords(x, y)
                    pc_val = self.pc.get_pc_at_location(0, x_new, y_new, PPCF=PPCF, xgrid=xgrid, ygrid=ygrid)
                    ax.text(x - 1.5, y - 1.5, f'{pc_val:.2f}', fontsize=8, color='black', alpha=0.8)

        # plot ball
        ax.plot(hometeam['ball_x'], hometeam['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
        return fig, ax
        

    def plot_pitchcontrol_for_frame(self, tracking_home_data, tracking_away_data, PPCF, pass_team, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (106.0,68)):
        """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )
        
        Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
        
        Parameters
        -----------
            event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
            events: Dataframe containing the event data
            tracking_home: (entire) tracking DataFrame for the Home team
            tracking_away: (entire) tracking DataFrame for the Away team
            PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
            alpha: alpha (transparency) of player markers. Default is 0.7
            include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
            annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
            field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
            
        NB: this function no longer requires xgrid and ygrid as an input
            
        Returrns
        -----------
        fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

        """    

        


        # plot frame and event
        fig,ax = self.plot_pitch(field_color='white', field_dimen = field_dimen)
        self.plot_frame( tracking_home_data, tracking_away_data, figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
        
        # plot pitch control surface
        if pass_team=='Home':
            cmap = 'bwr'
        else:
            cmap = 'bwr_r'
        ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

        return fig,ax


    def save_plot(self, fig, filename):
        fig.savefig(filename)

    def show_plot(self, fig):
        plt.show()




if __name__ == "__main__":

    from pitch_control import PitchControl

    


    #########################################################
    # PLOT PITCH CONTROL FOR A SPECIFIC FRAME
    #########################################################

    print("Plotting pitch control for a specific frame")

    match_id = 666385
    pitch_control = PitchControl(match_id)

    frame = 46125 
    frame = 12927 
    
    PPCF, xgrid, ygrid = pitch_control.run_pc_for_frame(frame)

    team_type = pitch_control.team_type


    visual = Visual(pitch_control)

    frame_home = pitch_control.home_tracking_data.iloc[frame]
    frame_away = pitch_control.away_tracking_data.iloc[frame]

    print(f"team_type {team_type}")

    fig, ax = visual.plot_pitchcontrol_for_frame(frame_home, frame_away, PPCF, team_type, annotate=True)

    visual.show_plot(fig)

    exit()


    #############################################################
    # PLOT PITCH CONTROL FOR A SPECIFIC FRAME USING IMPORTED DATA
    #############################################################

    print("Plotting pitch control for a specific frame using imported data")

    match_id = 666385
    frame_index = 46118
    frame_index = 46158
    
    data = np.load(f'{OUTPUTS_DIR}/{match_id}/PPCF_filtered_30x20.npz', allow_pickle=True)

    PPCF_data = data['PPCF_data']

    PPCF_data_list = PPCF_data.tolist()


    def get_PC_data_for_frame(frame_index):
        for frame in PPCF_data_list:
            if frame.get('frame_index') == frame_index:
                return frame
            
        print(f"No frame found for frame index {frame_index}")

        exit()
        return None


    frame_data = get_PC_data_for_frame(frame_index)

    print(frame_data.keys())
    print(frame_data.get('team_type'))

    pitch_control = PitchControl(match_id)

    PPCF = frame_data.get('PPCF')

    team_type = frame_data.get('team_type')

    pitch_control.PPCF_data = [frame_data]

    visual = Visual(pitch_control)

    frame_home = pitch_control.home_tracking_data.iloc[frame_index]
    frame_away = pitch_control.away_tracking_data.iloc[frame_index]

    fig, ax = visual.plot_pitchcontrol_for_frame(frame_home, frame_away, PPCF, team_type, annotate=True)

    visual.show_plot(fig)






        
        