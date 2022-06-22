from ast import Lambda
from requests import get
from pprint import PrettyPrinter


BASE_URL = 'https://data.nba.net'
ALL_JSON = '/prod/v1/today.json'

printer = PrettyPrinter()

def get_links():
    data = get(BASE_URL + ALL_JSON).json()
    links = data['links']
    return links


def get_scoreboard():
    scoreboard = get_links()['currentScoreboard']
    games = get(BASE_URL + scoreboard).json()['games']

    for game in games:
        home_team = game['hTeam']
        away_team = game['vTeam']
        clock = game['clock']
        period = game['period']['current']
        
        print('-'*40)
        print(f"{away_team['triCode']}: {away_team['score']} at {home_team['triCode']}: {home_team['score']}, {clock} remaining in period {period}")


def get_stats():
    stats = get_links()['leagueTeamStatsLeaders']
    teams = get(BASE_URL + stats).json()['league']['standard']['regularSeason']['teams']

    # Filter all-star teams and sort by rank
    teams = list(filter(lambda x: x['name'] != 'Team', teams))
    teams.sort(key=lambda x: int(x['ppg']['rank']))

    for rank, team in enumerate(teams):
        team_name = team['name']
        nickname = team['nickname']
        ppg = team['ppg']['avg']
        print(f'{rank+1:02d}. {team_name} {nickname}: {ppg}')


get_stats()