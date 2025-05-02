
import numpy as np
import scipy.signal as signal
from player import Player
import time

class MetricaPitchControl:
    def __init__(self):
        pass


    def to_metric_coordinates(self,data,field_dimen=(106.,68.) ):
        '''
        Convert positions from Metrica units to meters (with origin at centre circle)
        '''
        x_columns = [c for c in data.columns if c[-1].lower()=='x']
        y_columns = [c for c in data.columns if c[-1].lower()=='y']
        data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
        data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
        ''' 
        ------------ ***NOTE*** ------------
        Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
        I've changed the line above to reflect this. It was originally:
        data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
        ------------ ********** ------------
        '''
        return data
    

    def calc_player_velocities(self, team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
        """ calc_player_velocities( tracking_data )
        
        Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
        
        Parameters
        -----------
            team: the tracking DataFrame for home or away team
            smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
            filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
            window: smoothing window size in # of frames
            polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
            maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
            
        Returrns
        -----------
        team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

        """
        # remove any velocity data already in the dataframe
        team = self.remove_player_velocities(team)
        
        # Get the player ids
        player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

        # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
        dt = team['Time [s]'].diff()
        
        # index of first frame in second half
        second_half_idx = team.Period.idxmax()
        
        # estimate velocities for players in team
        for player in player_ids: # cycle through players individually
            # difference player positions in timestep dt to get unsmoothed estimate of velicity
            vx = team[player+"_x"].diff() / dt
            vy = team[player+"_y"].diff() / dt

            if maxspeed>0:
                # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
                raw_speed = np.sqrt( vx**2 + vy**2 )
                vx[ raw_speed>maxspeed ] = np.nan
                vy[ raw_speed>maxspeed ] = np.nan
                
            if smoothing:
                if filter_=='Savitzky-Golay':

                    # Get the first and second half data
                    first_half_data = vx.loc[:second_half_idx]
                    second_half_data = vx.loc[second_half_idx:]
                    
                    # Check if the window length is valid for both halves
                    if (len(first_half_data) >= window and window % 2 == 1) and (len(second_half_data) >= window and window % 2 == 1):
                        
                        # calculate first half velocity
                        vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                        vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                        # calculate second half velocity
                        vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                        vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)

                    else:
                        print(f"Window length is not valid for both halves for player {player}")
                        print(f"Window length: {window}")
                        print(f"First half length: {len(first_half_data)}")
                        print(f"Second half length: {len(second_half_data)}")
                        
                        
                elif filter_=='moving average':
                    ma_window = np.ones( window ) / window 
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                    vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                    vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                    
            
            # put player speed in x,y direction, and total speed back in the data frame
            team[player + "_vx"] = vx
            team[player + "_vy"] = vy
            team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

        return team

    def remove_player_velocities(self, team):
        # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
        columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
        team = team.drop(columns=columns)
        return team
    

    def default_model_params(self, time_to_control_veto=3):
        """
        default_model_params()
        
        Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
        
        Parameters
        -----------
        time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
        
        
        Returns
        -----------
        
        params: dictionary of parameters required to determine and calculate the model
        
        """
        # key parameters for the model, as described in Spearman 2018
        params = {}
        # model parameters
        params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
        params['max_player_speed'] = 5. # maximum player speed m/s
        params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
        params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
        params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
        params['lambda_att'] = 4.3 # ball control parameter for attacking team
        params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
        params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
        params['average_ball_speed'] = 15. # average ball travel speed in m/s
        # numerical parameters for model evaluation
        params['int_dt'] = 0.04 # integration timestep (dt)
        params['max_int_time'] = 10 # upper limit on integral time
        # params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location. ORIGINAL PARAMETER
        params['model_converge_tol'] = 0.1 # assume convergence when PPCF>0.99 at a given location.
        # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
        # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
        params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
        params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
        return params
    
    def find_goalkeeper(self, team):
        '''
        Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
        ''' 
        x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
        GK_col = team.iloc[0][x_columns].abs().idxmax()
        return GK_col.split('_')[1]
    
    def initialise_players(self,team,teamname,params,GKid):
        """
        initialise_players(team,teamname,params)
        
        create a list of player objects that holds their positions and velocities from the tracking data dataframe 
        
        Parameters
        -----------
        
        team: row (i.e. instant) of either the home or away team tracking Dataframe
        teamname: team name "Home" or "Away"
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
            
        Returns
        -----------
        
        team_players: list of player objects for the team at at given instant
        
        """    
        # get player  ids
        player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
        # create list
        team_players = []
        for p in player_ids:
            # create a player object for player_id 'p'
            
            team_player = Player(p,team,teamname,params,GKid)
            
            
            if team_player.inframe:
                team_players.append(team_player)
        return team_players
    
    def calculate_pitch_control_at_target(self, target_position, attacking_players, defending_players, ball_start_pos, params):
        """ calculate_pitch_control_at_target
        
        Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
        
        Parameters
        -----------
            target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
            attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
            defending_players: list of 'player' objects (see player class above) for the players on the defending team
            ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
            params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
            
        Returrns
        -----------
            PPCFatt: Pitch control probability for the attacking team
            PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

        """
        # calculate ball travel time from start position to end position.
        if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
            ball_travel_time = 0.0 
        else:
            # ball travel time is distance to target position from current ball position divided assumed average ball speed
            ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
        
        # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
        tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
        tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
        
        # check whether we actually need to solve equation 3
        if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
            # if defending team can arrive significantly before attacking team, no need to solve pitch control model
            return 0., 1.
        elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
            # if attacking team can arrive significantly before defending team, no need to solve pitch control model
            return 1., 0.
        else: 
            # solve pitch control model by integrating equation 3 in Spearman et al.
            # first remove any player that is far (in time) from the target location
            attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
            defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
            # set up integration arrays
            dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
            PPCFatt = np.zeros_like( dT_array )
            PPCFdef = np.zeros_like( dT_array )
            # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
            ptot = 0.0
            i = 1
            while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
                T = dT_array[i]
                for player in attacking_players:
                    # calculate ball control probablity for 'player' in time interval T+dt
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                    # make sure it's greater than zero
                    assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                    player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                    PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
                for player in defending_players:
                    # calculate ball control probablity for 'player' in time interval T+dt
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                    # make sure it's greater than zero
                    assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                    player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                    PPCFdef[i] += player.PPCF # add to sum over players in the defending team
                ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
                i += 1
            if i>=dT_array.size:
                print("Integration failed to converge: %1.3f" % (ptot) )
            return PPCFatt[i-1], PPCFdef[i-1]
            
    def check_offsides(self, attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
        """
        check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
        
        checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from 
        the 'attacking_players' list and ignored in the pitch control calculation.
        
        Parameters
        -----------
            attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
            defending_players: list of 'player' objects (see player class above) for the players on the defending team
            ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
            GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
            verbose: if True, print a message each time a player is found to be offside
            tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m
                
        Returrns
        -----------
            attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
        """    
        # find jersey number of defending goalkeeper (just to establish attack direction)
        defending_GK_id = GK_numbers[1] if attacking_players[0].teamname=='Home' else GK_numbers[0]
        # make sure defending goalkeeper is actually on the field!
        assert defending_GK_id in [p.id for p in defending_players], "Defending goalkeeper jersey number not found in defending players"
        # get goalkeeper player object
        defending_GK = [p for p in defending_players if p.id==defending_GK_id][0]  
        # use defending goalkeeper x position to figure out which half he is defending (-1: left goal, +1: right goal)
        defending_half = np.sign(defending_GK.position[0])
        # find the x-position of the second-deepest defeending player (including GK)
        second_deepest_defender_x = sorted( [defending_half*p.position[0] for p in defending_players], reverse=True )[1]
        # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
        offside_line = max(second_deepest_defender_x,defending_half*ball_position[0],0.0)+tol
        # any attacking players with x-position greater than the offside line are offside
        if verbose:
            for p in attacking_players:
                if p.position[0]*defending_half>offside_line:
                    print("player %s in %s team is offside" % (p.id, p.playername) )
        attacking_players = [p for p in attacking_players if p.position[0]*defending_half<=offside_line]
        return attacking_players
    

    def is_within_radius(self, target_position: np.ndarray, player_positions: np.ndarray, radius: float = 10.0) -> bool:
        """
        Check if the target position is within `radius` meters of any player position.
        
        Args:
            target_position (np.ndarray): shape (2,), the (x, y) of the target grid cell.
            player_positions (np.ndarray): shape (n, 2), array of all player positions.
            radius (float): distance threshold in meters.

        Returns:
            bool: True if target is within `radius` of any player, else False.
        """
        distances = np.linalg.norm(player_positions - target_position, axis=1)
        return np.any(distances <= radius)



    def generate_pitch_control_for_frame(self, tracking_home, tracking_away, params, GK_numbers, possesion_team = 'Home', field_dimen = (106.,68.,), n_grid_cells_x = 30, offsides=True):

        pass_team = possesion_team

        ball_start_pos = tracking_home.loc[['ball_x','ball_y']]


        # break the pitch down into a grid
        n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
        dx = field_dimen[0]/n_grid_cells_x
        dy = field_dimen[1]/n_grid_cells_y
        xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
        ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
        # initialise pitch control grids for attacking and defending teams 
        PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
        PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )


        # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
        if pass_team=='Home':
            attacking_players = self.initialise_players(tracking_home,'Home',params,GK_numbers[0])
            defending_players = self.initialise_players(tracking_away,'Away',params,GK_numbers[1])
        elif pass_team=='Away':
            defending_players = self.initialise_players(tracking_home,'Home',params,GK_numbers[0])
            attacking_players = self.initialise_players(tracking_away,'Away',params,GK_numbers[1])
        else:
            assert False, "Team in possession must be either home or away"

        

        
            
        # find any attacking players that are offside and remove them from the pitch control calculation
        if offsides:
            attacking_players = self.check_offsides( attacking_players, defending_players, ball_start_pos, GK_numbers)
        
        # calculate pitch pitch control model at each location on the pitch

        for i in range( len(ygrid) ):
            for j in range( len(xgrid) ):
                target_position = np.array( [xgrid[j], ygrid[i]] )
                start_time = time.time()

                player_positions = np.array([p.position for p in attacking_players + defending_players])


                # PPCFa[i,j], PPCFd[i,j] = self.calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params) 

                if self.is_within_radius(target_position, player_positions, radius=10):
                    PPCFa[i,j], PPCFd[i,j] = self.calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)

                else:
                    PPCFa[i,j], PPCFd[i,j] = 0.5, 0.5


                
    
    
        
        # # check probabilitiy sums within convergence
        # checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
        # assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)



        return PPCFa,xgrid,ygrid
        