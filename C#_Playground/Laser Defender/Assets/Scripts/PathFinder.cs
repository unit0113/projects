using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathFinder : MonoBehaviour
{
    EnemyFactory enemyFactory;
    WaveConfigSO waveConfig;
    List<Transform> waypoints;
    int waypointIndex = 0;

    void Awake() {
        enemyFactory = FindObjectOfType<EnemyFactory>();
        waveConfig = enemyFactory.GetCurrentWave();
    }

    void Start() {
        waypoints = waveConfig.GetWayPoints();
        transform.position = waypoints[waypointIndex].position;
    }

    void Update() {
        if (waypointIndex < waypoints.Count) {
            Vector3 targetPosition = waypoints[waypointIndex].position;
            float deltaMove = waveConfig.GetMoveSpeed() * Time.deltaTime;
            transform.position = Vector2.MoveTowards(transform.position, targetPosition, deltaMove);
            if (transform.position == targetPosition) { ++waypointIndex; }
        } else {
            Destroy(gameObject);
        }
    }
}
