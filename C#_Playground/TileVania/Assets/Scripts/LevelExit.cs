using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelExit : MonoBehaviour
{
    [SerializeField] float exitDelay = 1f;
    void OnTriggerEnter2D(Collider2D other) {
        if (other.tag == "Player") {
            StartCoroutine(LoadNextLevel());
        }
    }

    IEnumerator LoadNextLevel() {
        yield return new WaitForSecondsRealtime(exitDelay);
        int nextSceneIndex = SceneManager.GetActiveScene().buildIndex + 1;
        if (nextSceneIndex == SceneManager.sceneCountInBuildSettings) {
            nextSceneIndex = 0;
        }
        FindObjectOfType<ScenePersit>().ResetScenePersit();
        SceneManager.LoadScene(nextSceneIndex);
    }
}
