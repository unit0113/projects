using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelManager : MonoBehaviour
{
    [SerializeField] float sceneLoadDelay = 2f;
    ScoreKeeper scoreKeeper;

    void Awake() {
        scoreKeeper = FindObjectOfType<ScoreKeeper>();
    }
    public void LoadGame() {
        scoreKeeper.ResetScore();
        SceneManager.LoadScene("Game");
    }

    public void LoadMainMenu() {
        SceneManager.LoadScene("MainMenu");
    }

    public void LoadGameOver() {
        StartCoroutine(WaitAndLoad("GameOver", sceneLoadDelay));
    }

    public void QuitGame() {
        Application.Quit();
    }

    IEnumerator WaitAndLoad(string scene, float delay) {
        yield return new WaitForSecondsRealtime(delay);
        SceneManager.LoadScene(scene);
    }
}
