using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    [SerializeField] GameObject thingToFollow;
    Vector3 offset = new Vector3(0, 0, -10);
    void LateUpdate()
    {
        transform.position = thingToFollow.transform.position + offset;
    }
}
