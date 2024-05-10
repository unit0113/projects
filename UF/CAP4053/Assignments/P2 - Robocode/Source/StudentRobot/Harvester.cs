using Robocode;
using Robocode.Util;
using System.Drawing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;


namespace CAP4053.Student
{
    public class Harvester : TeamRobot
    {
        BlackBoard blackBoard;
        string currentTarget;
        string radarState;
        string movementState;
        int frame;
        int findCounter;
        Vector2 targetPos;
        int movementDirection;
        Random rnd;
        List<Tuple<string, int>> shotTracker;
        int radarSearchCounter;

        public Harvester()
        {
            blackBoard = new BlackBoard();
            rnd = new Random();
        }

        public override void Run() {
            // Set colors
            BodyColor = (Color.FromArgb(28, 59, 51));
            GunColor = (Color.FromArgb(113, 106, 78));
            RadarColor = (Color.FromArgb(180, 94, 51));
            ScanColor = (Color.FromArgb(78, 59, 53));
            BulletColor = (Color.FromArgb(108, 75, 54));

            // Independent usage of turret/radar
            IsAdjustRadarForGunTurn = true;
            IsAdjustGunForRobotTurn = true;

            blackBoard.clear();
            blackBoard.SetWidthHeight(BattleFieldHeight, BattleFieldWidth);
            currentTarget = "";
            frame = 0;
            findCounter = 10;
            targetPos = new Vector2((float)X, (float)Y);
            movementDirection = 1;
            shotTracker = new List<Tuple<string, int>>();
            radarSearchCounter = 0;

            // Set starting states
            radarState = "Search";
            movementState = "StopNGo";

            while (true) {
                ++frame;
                BroadcastMessage(new FriendlyInfo(Energy, X, Y, Name, Velocity, HeadingRadians));
                blackBoard.newFrame(frame);
                MoveIt();
                GunStuff();
                RadarStuff();
                blackBoard.endFrame();
                Execute();
            }
        }

        private void MoveIt()
        {
            // Revert to find state every so often
            --findCounter;
            if (findCounter < 0)
            {
                movementState = "Find";
                findCounter = 100;
            }

            // Update shot tracker
            blackBoard.UpdateShotTracker(shotTracker);

            if (movementState == "Find")
            {
                Vector2 newTargetPos = blackBoard.FindDesiredDistanceGrid(this);
                movementState = "StopNGo";
                if (newTargetPos != targetPos)
                {
                    targetPos = newTargetPos;
                    SetTurnRightRadians(Utils.NormalRelativeAngle(GetBearingTo(targetPos) - HeadingRadians));
                    movementState = "FindRepositionTransition";
                    
                }
            }

            if (movementState == "goToMiddle")
            {
                targetPos = new Vector2((float)BattleFieldWidth / 2, (float)BattleFieldHeight / 2);
                SetTurnRightRadians(Utils.NormalRelativeAngle(GetBearingTo(targetPos) - HeadingRadians));
                movementState = "FindRepositionTransition";
            }

            if (movementState == "FindRepositionTransition")
            {
                if (Math.Abs(TurnRemaining) < 1)
                {
                    SetAhead(GetDistanceTo(targetPos));
                    movementState = "Reposition";
                }
            }

            if (movementState == "Reposition")
            {
                // Wiggle, wiggle, wiggle, wiggle, wiggle, yeah
                if (Math.Abs(TurnRemaining) < 1)
                {
                    SetTurnRight(20 * movementDirection);
                    movementDirection *= -1;
                }

                // If desitination reached
                if (DistanceRemaining < 1) {
                    movementState = "StopNGo";
                    shotTracker.Clear();
                }
            }

            if (movementState == "StopNGo")
            {
                // Turn perpendicular to whatever shot the next arriving shot
                if (Math.Abs(TurnRemaining) < 1)
                {
                    if (shotTracker.Count > 0)
                    {
                        SetTurnRightRadians(Utils.NormalRelativeAngle(blackBoard.GetBestAngle(shotTracker[0].Item1)));
                    }
                    else if (currentTarget != "")
                    {
                        SetTurnRightRadians(Utils.NormalRelativeAngle(blackBoard.GetBestAngle(currentTarget)));
                    }
                }
                

                if (shotTracker.Count > 0 && frame > shotTracker[0].Item2 - 12)
                {
                    movementDirection = rnd.Next(0, 100) < 50 ? 1 : -1;
                    SetAhead(movementDirection * rnd.Next(40, 70));
                    shotTracker.Clear();
                }
            }
        }

        private void GunStuff()
        {
            // Update later with momentum and other checks
            currentTarget = blackBoard.getTarget();
            // Set radar state
            if (currentTarget == "")
            {
                radarState = "Search";
            } else {
                radarState = "TWS";
            }

            // Don't fire till you see the whites of their eyes!
            if (Energy < 25 && currentTarget != "")
            {
                blackBoard.SetDesiredDistance(currentTarget, 100);
            }

            // Exit if no target
            if (currentTarget == "") { return; }

            // Get shot power
            double shotPower = blackBoard.getShotPower(currentTarget);

            // Fire if current angle works
            if (GunHeat == 0 && blackBoard.PredictHit(currentTarget, shotPower, this)) {
                SetFire(shotPower);
                blackBoard.addShot(frame, currentTarget);
            }

            // Turn gun to new angle
            double firingAngle = blackBoard.GetFiringAngle(currentTarget, shotPower, this);
            SetTurnGunRightRadians(Utils.NormalRelativeAngle(firingAngle - GunHeadingRadians));
        }

        private void RadarStuff()
        {
            if (radarState == "Search")
            {
                ++radarSearchCounter;
                // Look everywhere, find target
                SetTurnRadarLeft(Rules.RADAR_TURN_RATE);

                // If out of range of target
                if (radarSearchCounter > 20)
                {
                    movementState = "goToMiddle";
                }
            } else if (radarState == "TWS")
            {
                radarSearchCounter = 0;
                // Scan around next tick predicted position
                double angleToTurn = Utils.NormalRelativeAngle(GetBearingTo(blackBoard.PredictTargetLoc(currentTarget, 1)) - RadarHeadingRadians);
                if (angleToTurn < 0) { angleToTurn -= Rules.RADAR_TURN_RATE_RADIANS / 2; }
                else { angleToTurn += Rules.RADAR_TURN_RATE_RADIANS / 2; }
                SetTurnRadarRightRadians(angleToTurn);
            } else if (radarState == "Track")
            {
                // Single target tracking (possibly not needed, TWS seems better in every way)
                SetTurnRadarRightRadians(Utils.NormalRelativeAngle(GetBearingTo(blackBoard.PredictTargetLoc(currentTarget, 1)) - RadarHeadingRadians));
            }
        }

        public override void OnScannedRobot(ScannedRobotEvent e)
        {
            if (IsTeammate(e.Name)) { return; }
            blackBoard.update(e, this);
            BroadcastMessage(new BotData(e, frame, this));

        }

        public override void OnMessageReceived(MessageEvent e)
        {
            if (e.Message.GetType() == typeof(FriendlyInfo))
            {
                blackBoard.update((FriendlyInfo)e.Message);
            } else if (e.Message.GetType() == typeof(BotData))
            {
                blackBoard.update((BotData)e.Message);
            }
        }

        public override void OnBulletHit(BulletHitEvent e)
        {
            blackBoard.BulletHit(e.VictimName, this);
        }

        public override void OnBulletMissed(BulletMissedEvent e)

        {
            if (currentTarget != "")
            {
                blackBoard.BulletMiss(currentTarget);
            }
        }

        public override void OnHitWall(HitWallEvent e)
        {
            movementDirection *= -1;
            SetAhead(100);
        }

        public override void OnHitByBullet(HitByBulletEvent e)
        {
            blackBoard.IncreaseDesiredDistance(currentTarget);
            findCounter -= 5;
        }

        public override void OnRobotDeath(RobotDeathEvent e)
        {
            blackBoard.UpdateDeath(e);
        }

        private double GetBearingTo(Vector2 targetVector2)
        {
            return Math.Atan2(targetVector2.X - X, targetVector2.Y - Y);
        }

        private double GetDistanceTo(Vector2 target)
        {
            return Math.Sqrt(Math.Pow(target.X - X, 2) + Math.Pow(target.Y - Y, 2));
        }

        [Serializable]
        internal class BotData
        {
            public BotData(ScannedRobotEvent e, int _frame, TeamRobot me)
            {
                frame = _frame;
                name = e.Name;
                double absoluteBearing = me.HeadingRadians + e.BearingRadians;
                x = me.X + Math.Sin(absoluteBearing) * e.Distance;
                y = me.Y + Math.Cos(absoluteBearing) * e.Distance;
                vel = e.Velocity;
                headingRad = e.HeadingRadians;
                distance = e.Distance;
                energy = e.Energy;
                bearingRad = e.BearingRadians;
            }

            public int frame { get; }
            public string name { get; }
            public double x { get; }
            public double y { get; }
            public double vel { get; }
            public double headingRad { get; }
            public double distance { get; }
            public double absoluteBearing { get; }
            public double energy { get; }
            public double bearingRad { get; }
        }

        internal class BlackBoard
        {
            internal int frame;
            internal Dictionary<string, List<BotData>> enemies;
            internal Dictionary<string, FriendlyInfo> friends;
            internal int maxEntries = 1000;
            internal List<string> currentBots;
            float width;
            float height;
            int consecutiveHits;
            int consecutiveMisses;
            float consecutiveHitBonus = 0.1f;
            float consecutiveMissPenalty = 0.1f;
            Dictionary<string, Dictionary<int, Vector2>> predictions;
            Dictionary<int, string> shots;
            Dictionary<string, double> desiredDistances;

            public BlackBoard()
            {
                enemies = new Dictionary<string, List<BotData>>();
                friends = new Dictionary<string, FriendlyInfo>();
                currentBots = new List<string>();
                predictions = new Dictionary<string, Dictionary<int, Vector2>>();
                shots = new Dictionary<int, string>();
                consecutiveHits = 0;
                consecutiveMisses = 0;
                desiredDistances = new Dictionary<string, double>();
            }

            public void SetWidthHeight(double BattleFieldHeight, double BattleFieldWidth)
            {
                height = (float)BattleFieldHeight;
                width = (float)BattleFieldWidth;
            }

            public void BulletHit(string target, TeamRobot me)
            {
                // Ignore if friendly fire (oops)
                if (me.IsTeammate(target)) { return; }

                ++consecutiveHits;
                consecutiveMisses = 0;
                if (!desiredDistances.ContainsKey(target))
                {
                    desiredDistances[target] = 410;
                } else
                {
                    desiredDistances[target] = Math.Min(600, desiredDistances[target] + 10);
                }
            }
            public void BulletMiss(string target)
            {
                consecutiveHits = 0;
                ++consecutiveMisses;
                desiredDistances[target] = Math.Max(100, desiredDistances[target] -25);
            }

            public void addShot(int frame, string target)
            {
                shots.Add(frame, target);
            }

            public void newFrame(int newFrame) {
                frame = newFrame;
                foreach ( KeyValuePair<string, List<BotData>> item in enemies )
                {
                    if ( item.Value.Count > maxEntries)
                    {
                        item.Value.RemoveAt( 0 );
                    }
                }

                // Clear predictions
                predictions.Clear();
            }

            public void endFrame()
            {
                // Remove old targets
                foreach (string key in enemies.Keys)
                {
                    if (currentBots.Contains(key) && enemies[key].Last().frame + 2 < frame)
                    {
                        currentBots.Remove(key);
                    }
                }
            }

            public void update(ScannedRobotEvent e, TeamRobot me)
            {
                // Ignore if teammate
                if (me.IsTeammate(e.Name)) { return; }

                // Add if data is current
                if (!enemies.ContainsKey(e.Name))
                {
                    enemies[e.Name] = new List<BotData>
                    {
                        new BotData(e, frame, me)
                    };
                } else if (enemies[e.Name].Last().frame < frame)
                {
                    enemies[e.Name].Add(new BotData(e, frame, me));
                }
                // Update currentBots
                currentBots.Add(e.Name);

                // Add to desired distances
                if (!desiredDistances.ContainsKey(e.Name))
                {
                    desiredDistances[e.Name] = 400;
                }
            }

            public void update(FriendlyInfo info)
            {
                friends[info.name] = info;
            }

            public void update(BotData info)
            {
                if (!enemies.ContainsKey(info.name) || enemies[info.name].Last().frame < frame)
                {
                    enemies[info.name].Add(info);
                }

                // Add to desired distances
                if (!desiredDistances.ContainsKey(info.name))
                {
                    desiredDistances[info.name] = 400;
                }
            }

            public void UpdateDeath(RobotDeathEvent e)
            {
                if (currentBots.Contains(e.Name))
                {
                    currentBots.Remove(e.Name);
                }

                if (friends.ContainsKey(e.Name))
                {
                    friends.Remove(e.Name);
                }
            }

            public string getTarget()
            {
                string target = "";
                double minDist = double.MaxValue;
                foreach (KeyValuePair<string, List<BotData>> item in enemies)
                {
                    if (item.Value.Last().frame + 5 > frame && item.Value.Last().distance < minDist)
                    {
                        target = item.Key;
                        minDist = item.Value.Last().distance;
                    }
                }
                return target;
            }

            public Vector2 PredictTargetLoc(string target, int ticks)
            {
                // Memoize
                if (predictions.ContainsKey(target))
                {
                    if (predictions[target].ContainsKey(ticks)) { return predictions[target][ticks]; }
                } else
                {
                    predictions[target] = new Dictionary<int, Vector2>();
                }

                List<BotData> data = enemies[target];
                Vector2 prediction;
                // Linear prediction if single detection
                if (data.Count < 3) { prediction = PredictTargetLocLinear(target, ticks); }

                // Circular prediction if not enough data for pattern matching
                else if (data.Count < ticks * 50) { prediction = PredictTargetLocCircular(target, ticks); }

                // Pattern Matching
                else { prediction = PredictTargetLocPattern(target, ticks); }
                predictions[target][ticks] = prediction;
                return prediction;
            }

            private Vector2 PredictTargetLocLinear(string target, int ticks, int index=-1)
            {
                BotData data;
                if (index == -1) { data = enemies[target].Last(); }
                else { data = enemies[target][index]; }
                float x = (float)(data.x + ticks * Math.Sin(data.headingRad) * data.vel);
                float y = (float)(data.y + ticks * Math.Cos(data.headingRad) * data.vel);

                // Clamp values for collisions
                x = Math.Max(18, Math.Min(x, width - 18));
                y = Math.Max(18, Math.Min(y, height - 18));

                return new Vector2(x, y);
            }

            private Vector2 PredictTargetLocCircular(string target, int ticks)
            {
                List<BotData> data = enemies[target];
                double deltaVel = (data.Last().vel - data[data.Count - 3].vel) / 2;
                double deltaTheta = (data.Last().headingRad - data[data.Count - 3].headingRad) / 2;
                
                // If straight line, use linear
                if (Math.Abs(deltaVel) < 0.1 && Math.Abs(deltaTheta) < 0.1) { return PredictTargetLocLinear(target, ticks); }

                // Predict target
                double x = data.Last().x;
                double y = data.Last().y;
                double vel = data.Last().vel;
                double heading = data.Last().headingRad;

                for (int i = 0; i < ticks; ++i)
                {
                    heading += deltaTheta;
                    vel = Math.Max(-Rules.MAX_VELOCITY, Math.Min(vel + deltaVel, Rules.MAX_VELOCITY));
                    x += Math.Sin(heading) * (vel);
                    y += Math.Cos(heading) * (vel);

                    // Clamp values for collisions
                    if (x < 18 || x > width - 18 || y < 18 || y > height - 18)
                    {
                        x = Math.Max(18, Math.Min(x, width - 18));
                        y = Math.Max(18, Math.Min(y, height - 18));
                        break;
                    }
                }

                return new Vector2((float)x, (float)y);
            }

            //https://robowiki.net/wiki/Pattern_Matching
            private Vector2 PredictTargetLocPattern(string target, int ticks)
            {
                // If short range, just use linear
                if (ticks < 5) { return PredictTargetLocLinear(target, ticks); }

                List<double> deltaVels = new List<double>();
                List<double> deltaThetas = new List<double>();
                List<BotData> data = enemies[target];

                // Find changes in vel and heading
                for (int i = 0; i < data.Count - 1; ++i)
                {
                    deltaVels.Add(data[i + 1].vel - data[i].vel);
                    deltaThetas.Add(data[i + 1].headingRad - data[i].headingRad);
                }
                
                int windowLen = ticks;
                List<double> windowVels = new List<double>();
                List<double> windowThetas = new List<double>();
                // Save window
                for (int i = deltaVels.Count - windowLen; i < deltaVels.Count; ++i)
                {
                    windowVels.Add(deltaVels[i]);
                    windowThetas.Add(deltaThetas[i]);
                }
                // Find closest window match
                double minMatchValVel = double.MaxValue;
                int minMatchIndexVel = 0;
                double minMatchValTheta = double.MaxValue;
                int minMatchIndexTheta = 0;
                double difference;
                bool endOnShot;

                for (int i = 0; i < data.Count - 2 * windowLen; ++i)
                {
                    // Continue if break in data
                    if (!isValidWindow(i, target, windowLen)) { continue; }
                    difference = Difference(windowVels, deltaVels.Skip(i).Take(windowLen).ToList());
                    // Acount for targets that react to shots
                    endOnShot = closeShot(i + windowLen - 1);
                    if (endOnShot) { difference /= 4; }
                    if (Math.Abs(difference) < minMatchValVel)
                    {
                        minMatchValVel = Math.Abs(difference);
                        minMatchIndexVel = i;
                    }

                    difference = Difference(windowThetas, deltaThetas.Skip(i).Take(windowLen).ToList());
                    // Acount for targets that react to shots
                    if (endOnShot) { difference /= 4; }
                    if (Math.Abs(difference) < minMatchValTheta)
                    {
                        minMatchValTheta = Math.Abs(difference);
                        minMatchIndexTheta = i;
                    }
                }

                // Predict via windows
                double x = data.Last().x;
                double y = data.Last().y;
                double vel = data.Last().vel;
                double heading = data.Last().headingRad;

                for (int i = 0; i < ticks; ++i)
                {
                    x += Math.Sin(heading + deltaThetas[minMatchIndexTheta + i + windowLen] / 2) * (vel + deltaVels[minMatchIndexVel + i + windowLen] / 2);
                    y += Math.Cos(heading + deltaThetas[minMatchIndexTheta + i + windowLen] / 2) * (vel + deltaVels[minMatchIndexVel + i + windowLen] / 2);
                    heading += deltaThetas[minMatchIndexTheta + i + windowLen];
                    vel += deltaVels[minMatchIndexVel + i + windowLen];

                    // Clamp velocity
                    vel = Math.Max(-Rules.MAX_VELOCITY, Math.Min(vel, Rules.MAX_VELOCITY));

                    // Clamp values for collisions
                    if (x < 18 || x > width - 18 || y < 18 || y > height - 18)
                    {
                        x = Math.Max(18, Math.Min(x, width - 18));
                        y = Math.Max(18, Math.Min(y, height - 18));
                        break;
                    }
                }
                return new Vector2((float)x, (float)y);
            }

            private bool isValidWindow(int strart, string target, int windowLen)
            {
                for (int i = 1; i < windowLen; ++i)
                {
                    if (enemies[target][i].frame != enemies[target][i-1].frame + 1) { return false; }
                }
                return true;
            }

            private bool closeShot(int endingFrame)
            {
                return (shots.ContainsKey(endingFrame) || shots.ContainsKey(endingFrame - 1) || shots.ContainsKey(endingFrame + 1));
            }

            private double Difference(List<double> l1, List<double> l2)
            {
                double sum = 0;
                for (int i = 0; i < l1.Count; ++i)
                {
                    sum += l1[i] - l2[i];
                }
                return sum;
            }

            public double getShotPower(string target)
            {
                BotData data = enemies[target].Last();
                double power;
                if (data.distance < 50) { power = 3; }
                else if (data.distance > 700) { power = 1; }
                else { power = 3 - (data.distance - 50) / (650) * (3 - 1); }
                // Adjust based on recent hits/misses
                return Math.Max(1, Math.Min(3, consecutiveHits * consecutiveHitBonus + power - consecutiveMisses * consecutiveMissPenalty));
            }

            public double GetFiringAngle(string target, double shotPower, TeamRobot me)
            {
                Vector2 prediction = PredictTargetLoc(target, TicksToIntercept(target, shotPower, me));
                Vector2 nextPos = GetNextPos(me);
                return Math.Atan2(prediction.X - nextPos.X, prediction.Y - nextPos.Y);
            }

            //https://robowiki.net/wiki/FuturePosition
            private Vector2 GetNextPos(TeamRobot me)
            {
                // If not moving
                if (me.DistanceRemaining == 0) { return new Vector2((float)me.X, (float)me.Y); }

                // Calc heading
                double turnRate = Math.Min(Rules.MAX_TURN_RATE, Rules.GetTurnRate(Math.Abs(me.Velocity))) * Math.PI / 180;
                double heading = me.HeadingRadians;
                if (me.TurnRemainingRadians > 0)
                {
                    if (me.TurnRemainingRadians < turnRate) { heading += me.TurnRemainingRadians;  }
                    else {  heading += turnRate; }
                } else if (me.TurnRemainingRadians < 0)
                {
                    if (me.TurnRemainingRadians > -turnRate) { heading += me.TurnRemainingRadians; }
                    else { heading -= turnRate; }
                }
                Utils.NormalAbsoluteAngle(heading);

                // Calc movement
                double velocity = getNewVelocity(me.Velocity, me.DistanceRemaining);

                // Clamp and split into components
                velocity = Math.Min(Rules.MAX_VELOCITY, Math.Max(-Rules.MAX_VELOCITY, velocity));
                double dx = Math.Sin(heading) * velocity;
                double dy = Math.Cos(heading) * velocity;
                return new Vector2((float)(me.X + dx), (float)(me.Y + dy));
            }

            //https://robowiki.net/wiki/User:Voidious/Optimal_Velocity#Hijack_2
            private double getNewVelocity(double currVel, double distance)
            {
                // Normalize out negatives
                if (distance < 0) { return -getNewVelocity(-currVel, -distance); }

                double decelTime = Math.Max(1, Math.Ceiling(Math.Sqrt((4*2/Rules.DECELERATION)*distance + 1) - 1) / 2);
                double decelDist = (decelTime / 2.0) * (decelTime - 1) * Rules.DECELERATION;
                double maxVel = ((decelTime - 1) * Rules.DECELERATION) + ((distance - decelDist) / decelTime);

                double goalVel = Math.Min(maxVel, Rules.MAX_VELOCITY);

                if (currVel >= 0)
                {
                    return Math.Max(currVel - Rules.DECELERATION, Math.Min(goalVel, currVel + Rules.ACCELERATION));
                } else
                {
                    return Math.Max(currVel - Rules.ACCELERATION, Math.Min(goalVel, currVel + maxDecel(-currVel)));
                }

            }

            //https://robowiki.net/wiki/User:Voidious/Optimal_Velocity#Hijack_2
            private double maxDecel(double speed)
            {
                return Math.Min(1, speed / Rules.DECELERATION) * Rules.DECELERATION + Math.Max(0, 1 - speed / Rules.DECELERATION) * Rules.ACCELERATION;
            }

            private int TicksToIntercept(string target, double shotPower, TeamRobot me)
            {
                int ticks = 0;
                Vector2 mePos = new Vector2((float)me.X, (float)me.Y);
                Vector2 prediction = PredictTargetLoc(target, ticks);
                while ((++ticks) * Rules.GetBulletSpeed(shotPower) < Vector2.Distance(prediction, mePos))
                {
                    prediction = PredictTargetLoc(target, ticks);
                }
                return ticks;
            }

            private int TicksToInterceptFriendly(string friend, double shotPower, TeamRobot me)
            {
                int ticks = 0;
                Vector2 mePos = new Vector2((float)me.X, (float)me.Y);
                Vector2 prediction = PredictFriendlyLocLinear(friend, ticks);
                while ((++ticks) * Rules.GetBulletSpeed(shotPower) < Vector2.Distance(prediction, mePos))
                {
                    prediction = PredictFriendlyLocLinear(friend, ticks);
                }
                return ticks;
            }

            private Vector2 PredictFriendlyLocLinear(string friend, int ticks)
            {
                FriendlyInfo data = friends[friend];

                float x = (float)(data.x + ticks * Math.Sin(data.heading) * data.velocity);
                float y = (float)(data.y + ticks * Math.Cos(data.heading) * data.velocity);

                // Clamp values for collisions
                x = Math.Max(18, Math.Min(x, width - 18));
                y = Math.Max(18, Math.Min(y, height - 18));

                return new Vector2(x, y);
            }

            public bool PredictFriendlyHit(string target, double shotPower, TeamRobot me)
            {
                // Check if there are any friendlies
                if (me.Teammates.Count() == 0) {  return false; }

                // Check if friendlies are all further away than target
                bool allFurther = true;
                int ticks = TicksToIntercept(target, shotPower, me);
                foreach (string friend in friends.Keys)
                {
                    if (TicksToInterceptFriendly(friend, shotPower, me) < ticks)
                    {
                        allFurther = false;
                    }
                }
                if (allFurther) { return false; }

                int targetBoxSize = 15;
                Vector2 prediction, previousPrediction, shotVectorEnd, shotVectorStart;

                for (int i = 1; i <= ticks; ++i)
                {
                    prediction = PredictFriendlyLocLinear(target, i);
                    previousPrediction = PredictFriendlyLocLinear(target, i - 1);

                    shotVectorEnd = new Vector2((float)(me.X + i * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                        (float)(me.Y + i * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                    shotVectorStart = new Vector2((float)(me.X + (i - 1) * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                          (float)(me.Y + (i - 1) * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                    if (VectorsIntersect(previousPrediction, prediction, shotVectorStart, shotVectorEnd)) { return true; }
                }


                // Account for lazy friends
                prediction = PredictFriendlyLocLinear(target, ticks);
                shotVectorEnd = new Vector2((float)(me.X + ticks * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                        (float)(me.Y + ticks * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                shotVectorStart = new Vector2((float)(me.X + (ticks - 1) * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                      (float)(me.Y + (ticks - 1) * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));

                double heading = enemies[target].Last().headingRad;
                Vector2 forward = new Vector2((float)(prediction.X + targetBoxSize * Math.Sin(heading)), (float)(prediction.Y + targetBoxSize * Math.Cos(heading)));
                Vector2 back = new Vector2((float)(prediction.X - targetBoxSize * Math.Sin(heading)), (float)(prediction.Y - targetBoxSize * Math.Cos(heading)));
                Vector2 left = new Vector2((float)(prediction.X + targetBoxSize * Math.Sin(heading - Math.PI / 2)), (float)(prediction.Y + targetBoxSize * Math.Cos(heading - Math.PI / 2)));
                Vector2 right = new Vector2((float)(prediction.X - targetBoxSize * Math.Sin(heading - Math.PI / 2)), (float)(prediction.Y - targetBoxSize * Math.Cos(heading - Math.PI / 2)));
                return (VectorsIntersect(forward, back, shotVectorStart, shotVectorEnd)
                        || VectorsIntersect(left, right, shotVectorStart, shotVectorEnd));

            }

            public bool PredictHit(string target, double shotPower, TeamRobot me)
            {
                // Avoid friendly fire
                if (PredictFriendlyHit(target, shotPower, me)) { return false; }

                int targetBoxSize = 15;
                int ticks = TicksToIntercept(target, shotPower, me);

                Vector2 prediction, previousPrediction, shotVectorEnd, shotVectorStart;
                for (int i = 1; i <= ticks; ++i)
                {
                    prediction = PredictTargetLoc(target, i);
                    previousPrediction = PredictTargetLoc(target, i - 1);

                    shotVectorEnd = new Vector2((float)(me.X + i * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                        (float)(me.Y + i * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                    shotVectorStart = new Vector2((float)(me.X + (i - 1) * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                          (float)(me.Y + (i - 1) * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                    if (VectorsIntersect(previousPrediction, prediction, shotVectorStart, shotVectorEnd)) { return true; }
                }

                // Account for static targets
                prediction = PredictTargetLoc(target, ticks);
                previousPrediction = PredictTargetLoc(target, ticks - 1);
                shotVectorEnd = new Vector2((float)(me.X + ticks * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                        (float)(me.Y + ticks * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));
                shotVectorStart = new Vector2((float)(me.X + (ticks - 1) * Rules.GetBulletSpeed(shotPower) * Math.Sin(me.GunHeading * Math.PI / 180)),
                                                      (float)(me.Y + (ticks - 1) * Rules.GetBulletSpeed(shotPower) * Math.Cos(me.GunHeading * Math.PI / 180)));

                double heading = enemies[target].Last().headingRad;
                Vector2 forward = new Vector2((float)(prediction.X + targetBoxSize * Math.Sin(heading)), (float)(prediction.Y + targetBoxSize * Math.Cos(heading)));
                Vector2 back = new Vector2((float)(prediction.X - targetBoxSize * Math.Sin(heading)), (float)(prediction.Y - targetBoxSize * Math.Cos(heading)));
                Vector2 left = new Vector2((float)(prediction.X + targetBoxSize * Math.Sin(heading - Math.PI / 2)), (float)(prediction.Y + targetBoxSize * Math.Cos(heading - Math.PI / 2)));
                Vector2 right = new Vector2((float)(prediction.X - targetBoxSize * Math.Sin(heading - Math.PI / 2)), (float)(prediction.Y - targetBoxSize * Math.Cos(heading - Math.PI / 2)));
                return (VectorsIntersect(forward, back, shotVectorStart, shotVectorEnd)
                        || VectorsIntersect(left, right, shotVectorStart, shotVectorEnd));
            }

            //https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
            private bool onSegment(Vector2 p, Vector2 q, Vector2 r)
            {
                if (q.X <= Math.Max(p.X, r.X) && q.X >= Math.Min(p.X, r.X) &&
                    q.Y <= Math.Max(p.Y, r.Y) && q.Y >= Math.Min(p.Y, r.Y))
                    return true;

                return false;
            }

            private int orientation(Vector2 p, Vector2 q, Vector2 r)
            {
                float val = (q.Y - p.Y) * (r.X - q.X) - (q.X - p.X) * (r.Y - q.Y);

                // Collinear
                if (Math.Abs(val) < 0.0001f) { return 0; }
                else if (val > 0) { return 1; }
                else { return 2; }
            }

            private bool VectorsIntersect(Vector2 p1, Vector2 q1, Vector2 p2, Vector2 q2)
            {
                int o1 = orientation(p1, q1, p2);
                int o2 = orientation(p1, q1, q2);
                int o3 = orientation(p2, q2, p1);
                int o4 = orientation(p2, q2, q1);

                // General case
                if (o1 != o2 && o3 != o4) { return true; }

                // Edge cases
                if (o1 == 0 && onSegment(p1, p2, q1)) { return true; }
                if (o2 == 0 && onSegment(p1, q2, q1)) { return true; }
                if (o3 == 0 && onSegment(p2, p1, q2)) { return true; }
                if (o4 == 0 && onSegment(p2, q1, q2)) { return true; }

                return false;
            }

            public void clear()
            {
                shots.Clear();
                friends.Clear();
                currentBots.Clear();
                consecutiveHits = 0;
                consecutiveMisses = 0;
            }

            public Vector2 FindMinIntensityGridCoords(TeamRobot me)
            {
                // Find current grid
                double gridWidth = width / 4;
                double gridHeight = height / 4;
                int gridCol = (int)Math.Floor(me.X / gridWidth);
                int gridRow = (int)Math.Floor(me.Y / gridHeight);

                // Divide arena into 16 zones, find lowest intensity zone
                double maxSafeness = 0; //find min intensity, high values are good
                int targetRow = gridRow;
                int targetCol = gridCol;
                double gridSafeness;
                foreach (int i in new List<int> { -1, 0, 1 })
                {
                    foreach (int j in new List<int> { -1, 0, 1 })
                    {
                        // Check if valid
                        if (0 <= gridRow + i && gridRow + i < 4 && 0 <= gridCol + j && gridCol + j < 4)
                        {
                            gridSafeness = calcGridIntensity(gridRow + i, gridCol + j);
                            if (gridSafeness > maxSafeness)
                            {
                                maxSafeness = gridSafeness;
                                targetRow = gridRow + i;
                                targetCol = gridCol + j;
                            }
                        }
                    }
                }
                float pixelX = targetCol * height / 4 + height / 8;
                float pixelY = targetRow * width / 4 + width / 8;
                return new Vector2(pixelX, pixelY);
            }

            private double calcGridIntensity(int row, int col)
            {
                double intensity = 0;
                int pixelX = (int)(col * height / 4 + height / 8);
                int pixelY = (int)(row * width / 4 + width / 8);
                foreach (string key in currentBots)
                {
                    intensity += Math.Sqrt(Math.Pow(pixelX - enemies[key].Last().x, 2) + Math.Pow(pixelY - enemies[key].Last().y, 2));
                }

                return intensity;
            }

            public Vector2 FindDesiredDistanceGrid(TeamRobot me)
            {
                // Find current grid
                double gridWidth = width / 4;
                double gridHeight = height / 4;
                int gridCol = (int)Math.Floor(me.X / gridWidth);
                int gridRow = (int)Math.Floor(me.Y / gridHeight);

                // Divide arena into 16 zones, find lowest intensity zone
                double bestOptimality = double.MaxValue; // Lower = better
                int targetRow = gridRow;
                int targetCol = gridCol;
                double gridOptimality;
                foreach (int i in new List<int> { -1, 0, 1 })
                {
                    foreach (int j in new List<int> { -1, 0, 1 })
                    {
                        // Check if valid
                        if (0 <= gridRow + i && gridRow + i < 4 && 0 <= gridCol + j && gridCol + j < 4)
                        {
                            gridOptimality = calcGridIntensity(gridRow + i, gridCol + j);
                            if (gridOptimality < bestOptimality)
                            {
                                bestOptimality = gridOptimality;
                                targetRow = gridRow + i;
                                targetCol = gridCol + j;
                            }
                        }
                    }
                }
                float pixelX = targetCol * height / 4 + height / 8;
                float pixelY = targetRow * width / 4 + width / 8;
                return new Vector2(pixelX, pixelY);
            }

            private double calcGridOptimality(int row, int col)
            {
                double optimality = 0;
                int pixelX = (int)(col * height / 4 + height / 8);
                int pixelY = (int)(row * width / 4 + width / 8);
                foreach (string key in currentBots)
                {
                    optimality += Math.Sqrt(Math.Pow(desiredDistances[key] - (pixelX - enemies[key].Last().x), 2) + Math.Pow(desiredDistances[key] - (pixelY - enemies[key].Last().y), 2));
                }

                return optimality;
            }

            public double GetBestAngle(string target)
            {
                return Utils.NormalAbsoluteAngle(Math.PI / 2 + enemies[target].Last().bearingRad);
            }

            public bool ShotDetected()
            {
                foreach (string key in currentBots)
                {
                    if (enemies[key].Count > 1
                        && enemies[key].Last().energy + 0.25 <= enemies[key][enemies[key].Count - 2].energy
                        && enemies[key].Last().energy + 3 >= enemies[key][enemies[key].Count - 2].energy)
                    {
                        return true;
                    }
                }
                return false;
            }

            public void UpdateShotTracker(List<Tuple<string, int>> shotTracker)
            {
                foreach (string key in currentBots)
                {
                    if (enemies[key].Count > 1
                        && enemies[key].Last().energy + 0.25 <= enemies[key][enemies[key].Count - 2].energy
                        && enemies[key].Last().energy + 3 >= enemies[key][enemies[key].Count - 2].energy)
                    {
                        double shotEnergy = enemies[key][enemies[key].Count - 2].energy - enemies[key].Last().energy;
                        double bulletVel = 20 - 3 * shotEnergy;
                        int framesToHit = (int)Math.Floor(enemies[key].Last().distance / bulletVel);
                        shotTracker.Add(new Tuple<string, int>(key, frame + framesToHit));
                    }
                }
            }

            public void IncreaseDesiredDistance(string target)
            {
                if (desiredDistances.ContainsKey(target))
                {
                    desiredDistances[target] = Math.Min(600, desiredDistances[target] + 25);
                }
                else
                {
                    desiredDistances[target] = 425;
                }
            }

            public void SetDesiredDistance(string target, double distance)
            {
                desiredDistances[target] = distance;
            }
        }

        // Message to pass friendly info
        [Serializable]
        internal struct FriendlyInfo
        {           
            public FriendlyInfo(double _energy, double _x, double _y, string _name, double _vel, double _heading)
            {
                energy = _energy;
                x = _x;
                y = _y;
                name = _name;
                velocity = _vel;
                heading = _heading;
            }
            
            public double energy { get; }
            public double x { get; }
            public double y { get; }
            public string name { get; }
            public double velocity { get; }
            public double heading { get; }
        }
    }
}
