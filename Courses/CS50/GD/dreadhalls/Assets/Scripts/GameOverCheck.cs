using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine.SceneManagement;
public class GameOverCheck : MonoBehaviour {
    public static Vector3 currentCameraPosition;
    private static ILogger logger = Debug.unityLogger;
    void start() {
        
    }

    private void Update() {
        currentCameraPosition = Camera.main.gameObject.transform.position;
       //logger.Log(currentCameraPosition.y);
        //Console.WriteLine(currentCameraPosition.y);
        if (!(currentCameraPosition.y < -20)) return;
        if (DontDestroy.instance) {
            DontDestroy.instance.GetComponents<AudioSource>()[0].Stop();
            DontDestroy.instance = null;
        }
        SceneManager.LoadScene("GameOver");
    }
}
