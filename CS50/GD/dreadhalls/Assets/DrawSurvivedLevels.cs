using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DrawSurvivedLevels : MonoBehaviour {
    public Text text;
    private static ILogger logger = Debug.unityLogger;
    // Start is called before the first frame update
    void Start() {
        text.text = string.Format(
            PlayerStats.currentLevel == 1 ? "You made it through {0} maze" : "You made it through {0} mazes.",
            PlayerStats.currentLevel);
    }

    // Update is called once per frame
    void Update() {

            
 
    }
}
