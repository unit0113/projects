using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

[CreateAssetMenu(menuName = "Wave Config", fileName = "New Wave Config")]
public class WaveConfigSO : ScriptableObject {
    [SerializeField] List<GameObject> enemiesPrefabs;
    [SerializeField] Transform pathPrefab;
    [SerializeField] float moveSpeed = 5f;
    [SerializeField] float timeBetweenSpawn = 1f;
    [SerializeField] float spawnTimeVariance = 0.25f;
    [SerializeField] float minSpawnTime = 0.2f;

    public float GetMoveSpeed() { return moveSpeed; }

    public Transform GetStartingWaypoint() { return pathPrefab.GetChild(0); }

    public List<Transform> GetWayPoints() {
        List<Transform> waypoints = new List<Transform>();
        foreach(Transform child in pathPrefab) {
            waypoints.Add(child);
        }
        return waypoints;
    }

    public int GetEnemyCount() { return enemiesPrefabs.Count; }

    public GameObject GetEnemyPrefab(int index) { return enemiesPrefabs[index]; }

    public float GetRandomSpawnTime() {
        float spawnTime = Random.Range(timeBetweenSpawn - spawnTimeVariance, timeBetweenSpawn + spawnTimeVariance);
        return Mathf.Max(spawnTime, minSpawnTime);
    }
}
