using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CurrentLevel : MonoBehaviour {
    public Text text;

    void start(){

    }

    // Update is called once per frame
    void Update(){
        text.text = string.Format("Current Maze:{0}", PlayerStats.currentLevel);
    }
}
