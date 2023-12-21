using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelOverScript : MonoBehaviour {
    private static ILogger logger = Debug.unityLogger;
    // Start is called before the first frame update
    void Start() {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnControllerColliderHit(ControllerColliderHit hit) {
        if (hit.gameObject.CompareTag("key")) {
            logger.Log("test");
            SceneManager.LoadScene("GameOver");
        }
    }
}
