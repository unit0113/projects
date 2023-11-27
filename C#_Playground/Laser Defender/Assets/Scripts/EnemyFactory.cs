using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyFactory : MonoBehaviour
{
    WaveConfigSO currentWave;
    [SerializeField] List<WaveConfigSO> waveConfigs;
    [SerializeField] float timeBetweenWaves = 2f;
    [SerializeField] bool wavesAreLooping = true;
    void Start() {
        StartCoroutine(SpawnEnemyWaves());
    }

    IEnumerator SpawnEnemyWaves() {
        do {
            foreach (WaveConfigSO wave in waveConfigs) {
                currentWave = wave;
                for (int i = 0; i < currentWave.GetEnemyCount(); ++i) {
                    Instantiate(currentWave.GetEnemyPrefab(i),
                            currentWave.GetStartingWaypoint().position,
                            Quaternion.Euler(0,0,180),
                            transform);
                    yield return new WaitForSecondsRealtime(currentWave.GetRandomSpawnTime());
                }
                yield return new WaitForSecondsRealtime(timeBetweenWaves);
            }
        } while (wavesAreLooping);
    }

    public WaveConfigSO GetCurrentWave() { return currentWave; }
}
