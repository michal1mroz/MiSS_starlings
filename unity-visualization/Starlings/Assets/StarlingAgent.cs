using System.Collections.Generic;
using System.Numerics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class StarlingAgent : Agent {
    public static List<StarlingAgent> AllAgents = new List<StarlingAgent>();

    [Header("Flocking Parameters")]
    public float worldSize = 20f;
    public float maxSpeed = 1f;
    public float neighborRadius = 3f;

    private UnityEngine.Vector3 velocity;

    private void Awake()
    {
        AllAgents.Add(this);
    }

    private void OnDestroy()
    {
        AllAgents.Remove(this);
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new UnityEngine.Vector3(
            Random.Range(0f, worldSize),
            0f,
            Random.Range(0f, worldSize)
        );
        velocity = new UnityEngine.Vector3(
            Random.Range(-0.5f, 0.5f),
            0f,
            Random.Range(-0.5f, 0.5f)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(velocity.x);
        sensor.AddObservation(velocity.z);

        UnityEngine.Vector3 avgRelPos = UnityEngine.Vector3.zero;
        UnityEngine.Vector3 avgRelVel = UnityEngine.Vector3.zero;
        int neighborCount = 0;

        UnityEngine.Vector3 myPos = transform.localPosition;

        foreach (var other in AllAgents)
        {
            if (other == this) continue;

            UnityEngine.Vector3 delta = other.transform.localPosition - myPos;
            float dist = delta.magnitude;
            if (dist < neighborRadius)
            {
                avgRelPos += delta;
                avgRelVel += other.velocity;
                neighborCount++;
            }
        }

        if (neighborCount > 0)
        {
            avgRelPos /= neighborCount;
            avgRelVel /= neighborCount;
        }

        sensor.AddObservation(avgRelPos.x);
        sensor.AddObservation(avgRelPos.z);

        sensor.AddObservation(avgRelVel.x);
        sensor.AddObservation(avgRelVel.z);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float dVx = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float dVz = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        velocity += new UnityEngine.Vector3(dVx, 0f, dVz) * 0.05f;

        if(velocity.magnitude > maxSpeed)
        {
            velocity = velocity.normalized * maxSpeed;
        }

        UnityEngine.Vector3 pos = transform.localPosition + velocity * Time.deltaTime;
        pos.x = Mathf.Repeat(pos.x, worldSize);
        pos.z = Mathf.Repeat(pos.z, worldSize);
        transform.localPosition = pos;
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, neighborRadius);
    }
}
