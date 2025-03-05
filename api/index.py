from flask import Flask, request, jsonify
import psycopg2
import pandas as pd
import numpy as np
from numpy.linalg import lstsq  # Replaced scipy.linalg with numpy.linalg
import itertools
import copy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Database credentials
DB_HOST = "scouting.frc971.org"
DB_PORT = 5000
DB_NAME = "postgres"
DB_USER = "tableau"
DB_PASSWORD = "xWYNKBkaHasO"

# ----- Data Processing -----

def fetch_and_process_data():
    """Fetch data from the database and process it into a list of dictionaries.
    
    Modified to group all coral data together, compute a base coral time,
    then adjust for each level using the mean coral level.
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    query = """
    SELECT *
    FROM stats2025 s
    WHERE s.comp_code IN ('2025camb', '2016nytr')
    ORDER BY s.match_number;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    results = []
    team_numbers = df['team_number'].unique()
    
    for team in team_numbers:
        team_data = df[df['team_number'] == team]
        A = []
        b = []
        # Build the regression matrix using three columns:
        # processor_teleop, the grouped coral (sum of l1-l4), and net_teleop (algae barge)
        for _, row in team_data.iterrows():
            processor = row['processor_teleop']
            coral = row['l1_teleop'] + row['l2_teleop'] + row['l3_teleop'] + row['l4_teleop']
            barge = row['net_teleop']
            A.append([processor, coral, barge])
            b.append(110)
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        try:
            solution, _, _, _ = lstsq(A, b)
            processor_time, base_coral_time, barge_time = solution
        except ValueError:
            processor_time = base_coral_time = barge_time = 0.0

        # Aggregate total coral counts per level across all matches for the team.
        total_l1 = team_data['l1_teleop'].sum()
        total_l2 = team_data['l2_teleop'].sum()
        total_l3 = team_data['l3_teleop'].sum()
        total_l4 = team_data['l4_teleop'].sum()
        total_coral = total_l1 + total_l2 + total_l3 + total_l4

        # Calculate the mean level that this team scored on.
        # (Weighted by the number of corals at each level.)
        if total_coral > 0:
            mean_level = (1 * total_l1 * total_l2  * total_l3 * total_l4) / total_coral
        else:
            mean_level = 0

        # Helper function to compute the multiplier for a given level.
        def compute_multiplier(level, mean_level):
            if level < mean_level:
                return 1 - 0.10 * (mean_level - level)
            elif level > mean_level:
                return 1 + 0.15 * (level - mean_level)
            else:
                return 1

        # Adjust base coral time for each level.
        # If the team never scored at that level (total count = 0), set time to 9999.
        if total_l1 == 0:
            l1_time = 9999
        else:
            l1_time = base_coral_time * compute_multiplier(1, mean_level)
        if total_l2 == 0:
            l2_time = 9999
        else:
            l2_time = base_coral_time * compute_multiplier(2, mean_level)
        if total_l3 == 0:
            l3_time = 9999
        else:
            l3_time = base_coral_time * compute_multiplier(3, mean_level)
        if total_l4 == 0:
            l4_time = 9999
        else:
            l4_time = base_coral_time * compute_multiplier(4, mean_level)

        # Compute average values for reporting purposes.
        avg_l1 = team_data['l1_teleop'].mean()
        avg_l2 = team_data['l2_teleop'].mean()
        avg_l3 = team_data['l3_teleop'].mean()
        avg_l4 = team_data['l4_teleop'].mean()
        avg_processor = team_data['processor_teleop'].mean()
        avg_barge = team_data['net_teleop'].mean()
        avg_defense = team_data['defense_time'].mean() if 'defense_time' in team_data.columns else 0.0

        # For consistency with your original logic, if processor or barge times are too low, set them to 9999.
        if processor_time < 1: processor_time = 9999
        if barge_time < 1: barge_time = 9999

        results.append({
            'team_number': team,
            'algae_processor': processor_time,
            'coral_l1': l1_time,
            'coral_l2': l2_time,
            'coral_l3': l3_time,
            'coral_l4': l4_time,
            'algae_barge': barge_time,
            'avg_l1': avg_l1,
            'avg_l2': avg_l2,
            'avg_l3': avg_l3,
            'avg_l4': avg_l4,
            'avg_processor': avg_processor,
            'avg_barge': avg_barge,
            'avg_defense': avg_defense
        })
    return results

# ----- Robot Class -----

class Robot:
    """Represents a robot with performance metrics for simulation."""
    def __init__(self, name, coral_times, algae_times, actual):
        self.name = name
        self.coral_times = coral_times  # Dict: {level: time}
        self.algae_times = algae_times  # Dict: {type: time}
        self.remaining_time = 150
        self.coral_cycles = {1: 0, 2: 0, 3: 0, 4: 0}
        self.algae_cycles = {'barge': 0, 'processor': 0}
        self.defense_time = 0
        self.points = 0
        self.actual = actual  # Dict: actual averages from data

# ----- Helper Functions -----

def create_robots_from_data(data):
    """Create Robot objects from processed data."""
    robots = []
    for row in data:
        name = row['team_number']
        coral_times = {
            1: row['coral_l1'],
            2: row['coral_l2'],
            3: row['coral_l3'],
            4: row['coral_l4'],
        }
        algae_times = {
            'barge': row['algae_barge'],
            'processor': row['algae_processor']
        }
        actual = {
            'l1': row['avg_l1'],
            'l2': row['avg_l2'],
            'l3': row['avg_l3'],
            'l4': row['avg_l4'],
            'barge': row['avg_barge'],
            'processor': row['avg_processor'],
            'defense': row['avg_defense']
        }
        robots.append(Robot(name, coral_times, algae_times, actual))
    return robots

def simulate_alliance(alliance):
    """Simulate an alliance's performance and return points and details."""
    alliance_robots = [copy.deepcopy(robot) for robot in alliance]
    global_coral = {1: 0, 2: 0, 3: 0, 4: 0}
    global_algae = 0
    while True:
        best_efficiency = 0
        best_robot = best_action = best_action_time = None
        for robot in alliance_robots:
            if robot.remaining_time <= 0:
                continue
            for level in [1, 2, 3, 4]:
                if global_coral[level] >= 12:
                    continue  # Max of 12 coral per level
                time_cost = robot.coral_times[level] + 0.5 * global_coral[level]
                if time_cost <= robot.remaining_time:
                    efficiency = (1 + level) / time_cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_robot, best_action, best_action_time = robot, ('coral', level), time_cost
            for algae_type in ['barge', 'processor']:
                time_cost = robot.algae_times[algae_type] + 0.3 * global_algae
                points = 4 if algae_type == 'barge' else 2
                if time_cost <= robot.remaining_time:
                    efficiency = points / time_cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_robot, best_action, best_action_time = robot, ('algae', algae_type), time_cost
            if robot.remaining_time >= 10:
                defense_cost = 10 if robot.defense_time == 0 else 1
                if 0.05 >= best_efficiency and defense_cost <= robot.remaining_time:
                    best_efficiency = 0.05
                    best_robot, best_action, best_action_time = robot, ('defense', None), defense_cost
        if not best_robot:
            break
        if best_action[0] == 'coral':
            level = best_action[1]
            best_robot.coral_cycles[level] += 1
            best_robot.points += 1 + level
            global_coral[level] += 1
        elif best_action[0] == 'algae':
            algae_type = best_action[1]
            best_robot.algae_cycles[algae_type] += 1
            best_robot.points += 4 if algae_type == 'barge' else 2
            global_algae += 1
        elif best_action[0] == 'defense':
            added_time = 10 if best_robot.defense_time == 0 else 1
            best_robot.defense_time += added_time
            best_robot.points += 0
        best_robot.remaining_time -= best_action_time
    total_points = sum(r.points for r in alliance_robots)
    details = {r.name: {
        "coral_l1": r.coral_cycles[1], "coral_l2": r.coral_cycles[2],
        "coral_l3": r.coral_cycles[3], "coral_l4": r.coral_cycles[4],
        "algae_barge": r.algae_cycles['barge'], "algae_processor": r.algae_cycles['processor'],
        "defense_time": r.defense_time, "total_points": r.points,
        "time_coral_l1": r.coral_times[1], "time_coral_l2": r.coral_times[2],
        "time_coral_l3": r.coral_times[3], "time_coral_l4": r.coral_times[4],
        "time_algae_barge": r.algae_times['barge'], "time_algae_processor": r.algae_times['processor']
    } for r in alliance_robots}
    return total_points, details


def aggregate_simulations(robots):
    """Aggregate simulation results for picklist generation."""
    team_agg = {}
    first_team = robots[0]
    for team_a, team_b in itertools.combinations(robots[1:], 2):
        alliance = (first_team, team_a, team_b)
        _, results = simulate_alliance(alliance)
        sorted_teams = sorted(results.items(), key=lambda x: -x[1]['total_points'])
        for rank, (name, metrics) in enumerate(sorted_teams, 1):
            weight = 1 / rank
            if name not in team_agg:
                team_agg[name] = {k: 0 for k in ["total_weight"] + [f"weighted_{m}" for m in metrics]}
            team_agg[name]['total_weight'] += weight
            for metric, value in metrics.items():
                team_agg[name][f'weighted_{metric}'] += weight * value
    return team_agg

def build_picklist(robots, team_agg):
    """Build the picklist data as a list of dictionaries."""
    actual_map = {r.name: r.actual for r in robots}
    picklist_data = []
    for name, data in team_agg.items():
        total_weight = data['total_weight']
        sim_avg = {m: data[f'weighted_{m}'] / total_weight for m in [
            'coral_l1', 'coral_l2', 'coral_l3', 'coral_l4', 'algae_barge', 'algae_processor', 'defense_time'
        ]}
        actual = actual_map.get(name, {k: 0 for k in ['l1', 'l2', 'l3', 'l4', 'barge', 'processor', 'defense']})
        row = {'name': name, 'weighted_score': total_weight}
        for m in ['l1', 'l2', 'l3', 'l4']:
            row[f'sim_avg_{m}'] = sim_avg[f'coral_{m}']
            row[f'actual_{m}'] = actual[m]
            row[f'pct_diff_{m}'] = (sim_avg[f'coral_{m}'] - actual[m]) / actual[m] * 100 if actual[m] else 'N/A'
        for m in ['barge', 'processor']:
            row[f'sim_avg_{m}'] = sim_avg[f'algae_{m}']
            row[f'actual_{m}'] = actual[m]
            row[f'pct_diff_{m}'] = (sim_avg[f'algae_{m}'] - actual[m]) / actual[m] * 100 if actual[m] else 'N/A'
        row['sim_avg_defense'] = sim_avg['defense_time']
        row['actual_defense'] = actual['defense']
        row['pct_diff_defense'] = (sim_avg['defense_time'] - actual['defense']) / actual['defense'] * 100 if actual['defense'] else 'N/A'
        picklist_data.append(row)
    return picklist_data

# ----- Flask Routes -----

@app.route("/picklist", methods=["GET"])
def picklist():
    """Generate and return the picklist as JSON."""
    try:
        optimized_data = fetch_and_process_data()
        robots = create_robots_from_data(optimized_data)
        if len(robots) < 3:
            return "Insufficient teams for simulation.", 400
        team_agg = aggregate_simulations(robots)
        picklist_data = build_picklist(robots, team_agg)
        return jsonify(picklist_data)
    except Exception as e:
        return f"Error generating picklist: {e}", 500

@app.route("/simulate", methods=["POST"])
def simulate():
    """Simulate a match between two alliances and return results as JSON."""
    try:
        data = request.get_json()
        alliance1_teams = data.get("alliance1", [])
        alliance2_teams = data.get("alliance2", [])
        if len(alliance1_teams) != 3 or len(alliance2_teams) != 3:
            return "Each alliance must have exactly 3 teams.", 400
        optimized_data = fetch_and_process_data()
        robots = create_robots_from_data(optimized_data)
        robot_dict = {robot.name: robot for robot in robots}
        alliance1 = [robot_dict[team] for team in alliance1_teams if team in robot_dict]
        alliance2 = [robot_dict[team] for team in alliance2_teams if team in robot_dict]
        if len(alliance1) != 3 or len(alliance2) != 3:
            return "One or more teams not found.", 404
        points1, details1 = simulate_alliance(alliance1)
        points2, details2 = simulate_alliance(alliance2)
        # Adjust points for defense impact
        total_defense_time1 = sum(details1[robot.name]['defense_time'] for robot in alliance1)
        total_defense_time2 = sum(details2[robot.name]['defense_time'] for robot in alliance2)
        points2 -= (total_defense_time1 / 150) * points2 * 0.05
        points1 -= (total_defense_time2 / 150) * points1 * 0.05
        result = {
            "alliance1_points": points1,
            "alliance2_points": points2,
            "alliance1_details": details1,
            "alliance2_details": details2
        }
        return jsonify(result)
    except Exception as e:
        return f"Error simulating match: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
