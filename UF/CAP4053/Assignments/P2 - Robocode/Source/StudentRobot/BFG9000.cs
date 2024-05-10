using Robocode;
using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Drawing;
using System.Runtime.CompilerServices;
using Robocode.Util;
using System.Runtime;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics.Eventing.Reader;
using System.Reflection;
using System.Data;
using System.Diagnostics;
using Microsoft.SqlServer.Server;
using static CAP4053.Student.BFG9000;
using System.Drawing.Printing;
using System.Runtime.Serialization;

namespace CAP4053.Student
{
    public class BFG9000 : TeamRobot
    {
        public BehaviorTree roboTree;
        public double scanDirection = 1;
        public double moveDirection = 1;
        public bool hitWall = false;
        public bool hitByRamming = false;
        public int frame;
        public BotManager botManager = new BotManager();
        public EnemyBot enemy = new EnemyBot();

        public override void Run()
        {
            BodyColor = (Color.FromArgb(28, 59, 51));
            GunColor = (Color.FromArgb(113, 106, 78));
            RadarColor = (Color.FromArgb(180, 94, 51));
            ScanColor = (Color.FromArgb(78, 59, 53));
            BulletColor = (Color.FromArgb(108, 75, 54));

            SetColors(BodyColor, GunColor, RadarColor, BulletColor, ScanColor);

            BehaviorNode root = BuildTree(this, enemy);
            roboTree = new BehaviorTree(root, this, enemy);
     
            enemy.EnemyReset();
            while (true)
            {
                ++frame;
                BroadcastMessage(new FriendlyInfo(frame, Energy, X, Y, Name, Velocity, HeadingRadians));
                roboTree.Process();
                Execute();
            }           
        }

        // Bot methods 
        public override void OnScannedRobot(ScannedRobotEvent evnt)
        {
            // Return if scanned robot is friendly
            if (IsTeammate(evnt.Name))
            {
                return;
            }

            // if no enemy found or target or null
            if (enemy == null || enemy.NoTarget() || evnt.Name == enemy.GetName())
            {
                enemy.EnemyUpdate(evnt, this);                                     
            }
            BroadcastMessage(new BotData(evnt, frame, this));                               
            roboTree.Process();
        }
        public override void OnBulletMissed(BulletMissedEvent evnt)
        {
            //missInfo = evnt;
        }
        public override void OnHitWall(HitWallEvent evnt)
        {
            hitWall = true;
        }
        public override void OnHitByBullet(HitByBulletEvent evnt) { }
        public override void OnHitRobot(HitRobotEvent evnt)
        {
            hitByRamming = true;
        }
        public override void OnBulletHit(BulletHitEvent evnt) { }

        public override void OnRobotDeath(RobotDeathEvent evnt)
        {
            if (enemy != null && enemy.GetName().Equals(evnt.Name))
            {
                botManager.EnemyDied(evnt.Name);
                enemy.EnemyReset();
            }
            var friendly = botManager.GetFriendly(evnt.Name);
            if (friendly != null)
            {
                botManager.FriendlyDied(evnt.Name);
            }
        }
        public void UpdateIsAdjustGunForRobotTurn(bool turnOn)
        {
            IsAdjustGunForRobotTurn = turnOn;
        }
        public void UpdateIsAdjustRadarForGunTurn(bool turnOn)
        {
            IsAdjustRadarForGunTurn = turnOn;
        }
        public override void OnMessageReceived(MessageEvent evnt)
        {
            if (evnt.Message.GetType() == typeof(FriendlyInfo))
            {
                botManager.UpdateFriendly((FriendlyInfo)evnt.Message);
            }
            else if (evnt.Message.GetType() == typeof(BotData))
            {
                botManager.UpdateEnemy((BotData)evnt.Message);
            }
        }
        public void BroadcastMesssage(BotData botData)
        {
            BroadcastMessage(botData);
        }

        // helper functions for calculations
        public static double GetDistance(double x1, double y1, double x2, double y2)
        {
           return  Math.Sqrt(Math.Pow(x1 - x2, 2) + Math.Pow(y1 - y2, 2));
        }
        public static bool CheckOutOfBounds(double x, double y, double width, double height, double margin)
        {
            return x < margin || y < margin || x > width - margin || y > height - margin;
        }

        // Team Messaging classes and structs for storing message data
        [Serializable]
        public class BotData
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
        [Serializable]
        public struct FriendlyInfo
        {
            public FriendlyInfo(int _frame, double _energy, double _x, double _y, string _name, double _vel, double _heading)
            {
                frame = _frame;
                energy = _energy;
                x = _x;
                y = _y;
                name = _name;
                velocity = _vel;
                heading = _heading;
            }

            public int frame { get; }
            public double energy { get; }
            public double x { get; }
            public double y { get; }
            public string name { get; }
            public double velocity { get; }
            public double heading { get; }
        }
        public class BotManager
        {
            private Dictionary<string, EnemyBot> enemies = new Dictionary<string,EnemyBot>();
            private Dictionary<string, FriendlyInfo> friendlies = new Dictionary<string, FriendlyInfo>();

            public void UpdateEnemy(BotData data)
            {
                if (enemies.ContainsKey(data.name))
                {
                    enemies[data.name].EnemyData(data);

                }
                else
                {
                    EnemyBot newEnemy = new EnemyBot(data);
                    enemies.Add(data.name, newEnemy);
                }
            }

            //Update Friendlies
            public void UpdateFriendly(FriendlyInfo data)
            {
                friendlies[data.name] = data;
            }

            public EnemyBot GetEnemy(string name)
            {
                if(enemies.TryGetValue(name, out EnemyBot enemy))
                {
                    return enemy;
                }
                return null;
            }

            public FriendlyInfo? GetFriendly(string name)
            {
                if (friendlies.TryGetValue(name, out FriendlyInfo friend))
                {
                    return friend;
                }
                return null;
            }

            public IEnumerable<FriendlyInfo> GetFriendlies()
            {
                return friendlies.Values;
            }
            // Get all enemies
            public IEnumerable<EnemyBot> GetAllEnemies()
            {
                return enemies.Values;
            }

            public void EnemyDied(string name)
            {
                enemies.Remove(name);
            }
            public void FriendlyDied(string name)
            {
                friendlies.Remove(name);
            }
        }

        // References creation of enemy bot class for storing target information
        //https://mark.random-article.com/robocode/enemy_bot.html
        public class EnemyBot
        {
            private double x;
            private double y;
            private double bearing;
            private double distance;
            private double energy;
            private double heading;
            private double velocity;
            private string name;

            private double bearingRadians;
            private double headingRadians;

            private double prevHeadingRadians = 0.0;
            private double prevEnergy = 100.0;
            private double prevDistance = 0.0;

            public EnemyBot() { EnemyReset(); }

            public EnemyBot(BotData data)
            {
                x = data.x;
                y = data.y;
                velocity = data.vel;
                headingRadians = data.headingRad;
                energy = data.energy;
                distance = data.distance;
                bearingRadians = data.bearingRad;

            }
            public double GetX()
            {
                return this.x;
            }
            public double GetY()
            {
                return this.y;
            }
            public double GetBearing()
            {
                return this.bearing;
            }
            public double GetDistance()
            {
                return this.distance;
            }
            public double GetEnergy()
            {
                return this.energy;
            }
            public double GetHeading()
            {
                return this.heading;
            }
            public string GetName()
            {
                return this.name;
            }
            public double GetVelocity()
            {
                return this.velocity;
            }
            public double GetBearingRadians()
            {
                return this.bearingRadians;
            }
            public double GetHeadingRadians()
            {
                return this.headingRadians;
            }
            public double GetPrevHeadingRadians()
            {
                return this.prevHeadingRadians;
            }
            public double GetPrevEnergy()
            {
                return this.prevEnergy;
            }
            public double GetPrevDistance()
            {
                return this.prevDistance;
            }
            public void EnemyUpdate(ScannedRobotEvent e, TeamRobot me)
            {

                double absoluteBearing = me.HeadingRadians + e.BearingRadians;
                x = me.X + Math.Sin(absoluteBearing) * e.Distance;
                y = me.Y + Math.Cos(absoluteBearing) * e.Distance;

                // track previous values for preditions in targeting
                prevHeadingRadians = headingRadians;
                prevEnergy = energy;
                prevDistance = distance;


                // update enemy bot info from latest event
                distance = e.Distance;
                bearing = e.Bearing;
                energy = e.Energy;
                heading = e.Heading;
                velocity = e.Velocity;
                name = e.Name;
                bearingRadians = e.BearingRadians;
                headingRadians = e.HeadingRadians;
            }
            public void EnemyData(BotData data)
            {
                x = data.x;
                y = data.y;
                velocity = data.vel;
                headingRadians = data.headingRad;
                energy = data.energy;
                distance = data.distance;
                bearingRadians = data.bearingRad;
            }
            public void EnemyReset()
            {
                x = 0.0;
                y = 0.0;
                distance = 0.0;
                bearing = 0.0;
                energy = 0.0;
                heading = 0.0;
                velocity = 0.0;
                name = "";
                bearingRadians = 0.0;
                headingRadians = 0.0;
                prevHeadingRadians = 0.0;
                prevEnergy = 0.0;
                prevDistance = 0.0;
            }
            public bool NoTarget()
            {
                return string.IsNullOrEmpty(name);
            }
            public bool TargetAcquired()
            {
                return !(string.IsNullOrEmpty(name)) && energy > 0;
            }
        }

        // Behavior Tree implementation with Behavior Nodes
        // Composite Types Sequence and Selector
        // References used from ex0 assignment of Wumpus World and Behavior Tree Slides
        // Additional References
        // https://hub.packtpub.com/building-your-own-basic-behavior-tree-tutorial/
        // https://codereview.stackexchange.com/questions/109747/saving-and-resuming-position-while-iterating-over-a-container
        public class BehaviorTree
        {
            private readonly BehaviorNode root;

            public enum Status
            {
                Success,
                Failure,
                Running
            }
            public BehaviorTree(BehaviorNode behaviorRoot, BFG9000 bot, EnemyBot enemy)
            {
                this.root = behaviorRoot; 
            }
            public Status Process()
            {
                if (root == null)
                {
                    return Status.Failure;
                }
                else
                {
                    return root.Process();
                }
            }
        }
        private BehaviorNode BuildTree(BFG9000 bot, EnemyBot enemy)
        {
            Selector rootSelector = new Selector(bot, enemy);

            Selector searchEngageSelector = new Selector(bot, enemy);
            Sequence evadeSequence = new Sequence(bot, enemy);
            Selector statusSelector = new Selector(bot, enemy);
            Selector battleSelector = new Selector(bot, enemy);
            Sequence attackSequence = new Sequence(bot, enemy);
            Sequence patrolSequence = new Sequence(bot, enemy);
            Selector moveSelector = new Selector(bot, enemy);
            Sequence engageSequence = new Sequence(bot, enemy);
            BehaviorNode limitWideScan = new LimitNode(bot, enemy, new WideScan(bot, enemy), 2);
            Sequence allClearShoot = new Sequence(bot, enemy);
            Sequence friendlyFire = new Sequence(bot, enemy);
            Selector possibleToShoot = new Selector(bot, enemy);
            
            // Check bot position near wall, move away
            statusSelector.AddChild(new IsNearWall(bot, enemy, new AvoidWall(bot, enemy)));

            // Search for target through scanning if no current target
            // start target engage process if target available
            searchEngageSelector.AddChild(new NoTargets(bot, enemy, limitWideScan));
            searchEngageSelector.AddChild(engageSequence);

            // start engagement with enemy, confirm first that target is found
            engageSequence.AddChild(new HaveTarget(bot, enemy));
            engageSequence.AddChild(new AimRadar(bot, enemy));
            engageSequence.AddChild(battleSelector);

            // battle management, decide when to evade if rammed or attack
            battleSelector.AddChild(evadeSequence);
            battleSelector.AddChild(attackSequence);

            // Evade sequence for when enemy rams into bot, move back and start attack sequence          
            evadeSequence.AddChild(new RammedByEnemy(bot, enemy, new MoveBack(bot, enemy)));
            evadeSequence.AddChild(attackSequence);

            // attack sequence for when an enemy is targeted, lock onto target
            // adjust tank to face target
            // go through move sequence, if enemy is close or far
            // aim gun then go through shooting sequence to determine if friendlies are around
            attackSequence.AddChild(new ResetLimit(bot, enemy, limitWideScan));
            attackSequence.AddChild(new TargetLockRadar(bot, enemy));
            attackSequence.AddChild(new AdjustTank(bot, enemy));
            attackSequence.AddChild(moveSelector);
            attackSequence.AddChild(new AimGun(bot, enemy));
            attackSequence.AddChild(possibleToShoot);
            
            // Selector for choosing when to shoot
            possibleToShoot.AddChild(friendlyFire);
            possibleToShoot.AddChild(allClearShoot);

            // Friendly Fire sequence if there is a friendly in the way, move away then shoot
            friendlyFire.AddChild(new FriendlyInPath(bot, enemy, botManager));
            friendlyFire.AddChild(new GetClearShot(bot, enemy));
            friendlyFire.AddChild(new AimGun(bot, enemy));
            friendlyFire.AddChild(new FireGun(bot, enemy));

            // No friendlies in way or solo battle, fire gun
            allClearShoot.AddChild(new FireGun(bot, enemy));


            //patrol around if no targets
            patrolSequence.AddChild(new NoTargets(bot, enemy, new MoveForward(bot, enemy)));
            patrolSequence.AddChild(new WideScan(bot, enemy));


            // move towards bot if he moves far away
            moveSelector.AddChild(new EnemyOutRange(bot, enemy, new MoveTowardsEnemy(bot,enemy)));
            moveSelector.AddChild(new MoveRobot(bot, enemy));

            //Root Selector for selecting behavior branches
            rootSelector.AddChild(statusSelector);
            rootSelector.AddChild(searchEngageSelector);
            rootSelector.AddChild(patrolSequence);
            
            return rootSelector;
        }
        public abstract class BehaviorNode
        {
            protected List<BehaviorNode> children = new List<BehaviorNode>();
            protected BFG9000 myRobot;
            protected EnemyBot myEnemy;
            protected List<FriendlyInfo> friendlies = new List<FriendlyInfo>();
            protected BotManager myBotManager;
            
            public BehaviorNode(BFG9000 bot, EnemyBot enemy)
            {
                myRobot = bot;
                myEnemy = enemy;               
            }
            public BehaviorNode(BFG9000 bot, EnemyBot enemy, BotManager allBots)
            {
                myRobot = bot;
                myEnemy = enemy;
                myBotManager = allBots;
            }
            public void AddChild(BehaviorNode node)
            {
                children.Add(node);
            }
            public abstract BehaviorTree.Status Process();
        }
        public class Sequence : BehaviorNode
        {
            public Sequence(BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                foreach (var child in this.children)
                {
                    BehaviorTree.Status result = child.Process();
                    if (result == BehaviorTree.Status.Failure)
                    {
                        return BehaviorTree.Status.Failure;
                    }
                    if (result == BehaviorTree.Status.Running)
                    {
                        return BehaviorTree.Status.Running;
                    }
                }
                return BehaviorTree.Status.Success;
            }
        }
        public class Selector : BehaviorNode
        {
            public Selector (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                foreach (var child in this.children)
                {
                    BehaviorTree.Status result = child.Process();
                    if (result == BehaviorTree.Status.Success)
                    {
                        return BehaviorTree.Status.Success;
                    }
                    if (result == BehaviorTree.Status.Running)
                    {
                        return BehaviorTree.Status.Running;
                    }
                }
                return BehaviorTree.Status.Failure;
            }
        }
        public abstract class ConditionalNode : BehaviorNode
        {
            protected BehaviorNode child;
            public ConditionalNode (BFG9000 bot, EnemyBot enemy, BehaviorNode childNode) : base (bot, enemy)
            {
                this.child = childNode;
            }
            public ConditionalNode (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) 
            {
                this.child = null;
            }
            public ConditionalNode(BFG9000 bot, EnemyBot enemy, BotManager manager) : base(bot, enemy, manager)
            {
                this.child = null;
            }
            public ConditionalNode(BFG9000 bot, EnemyBot enemy, BotManager manager, BehaviorNode childNode) : base(bot, enemy, manager)
            {
                this.child = childNode;
            }
            public override BehaviorTree.Status Process()
            {
                if (CheckCondition())
                {
                    if(child != null)
                    {
                        return child.Process();
                    }
                    return BehaviorTree.Status.Success;                   
                }
                else
                {
                    return BehaviorTree.Status.Failure;
                }
            }
            protected abstract bool CheckCondition();
        }
        public abstract class DecoratorNode : BehaviorNode
        {
            protected BehaviorNode child;
            public DecoratorNode (BFG9000 bot, EnemyBot enemy, BehaviorNode childNode) : base (bot, enemy)
            {
                this.child = childNode;
            }
            public override BehaviorTree.Status Process()
            {
                if (CanExecute())
                {
                    return child.Process();
                }
                else
                {
                    HandleFailure();
                    return BehaviorTree.Status.Failure;
                }
            }

            protected abstract bool CanExecute();
            protected virtual void HandleFailure()
            {
            }
        }
        public class DebugNode : DecoratorNode
        {
            private string msg;
            public DebugNode(BFG9000 bot, EnemyBot enemy, BehaviorNode child, string log) : base(bot, enemy, child)
            {
                this.msg = log;
            }
            public override BehaviorTree.Status Process()
            {
                Console.WriteLine("Enter Node: " + msg);

                var status = child.Process();

                Console.WriteLine("Node processed: " + msg + "with status" + status);
                return status;
            }
            protected override bool CanExecute() 
            {
                return true;
            }
        }
        public class LimitNode : DecoratorNode
        {
            private readonly int limit;
            private int count;
            public LimitNode (BFG9000 bot, EnemyBot enemy, BehaviorNode child, int maxExecutions) : base (bot, enemy, child)
            {
                this.limit = maxExecutions;
                this.count = 0;
            }
            protected override bool CanExecute()
            {               
                return count < limit;
            }
            public override BehaviorTree.Status Process()
            {
                if (CanExecute())
                {
                    var result = child.Process();
                    //count++;
                    if (result == BehaviorTree.Status.Success)
                    {                        
                        count++;
                    }
                    return result;
                }
                else
                {
                    return BehaviorTree.Status.Failure;
                }
            }
            public void ResetCount()
            {
                count = 0;
            }
        }
        public class ResetLimit : DecoratorNode
        {
            public ResetLimit(BFG9000 bot, EnemyBot enemy, BehaviorNode child) : base(bot, enemy, child) {}

            protected override bool CanExecute()
            {
                return true;
            }
            public override BehaviorTree.Status Process()
            {
                BehaviorTree.Status result = child.Process();

                if (CanExecute())
                {
                    if (result == BehaviorTree.Status.Success || result == BehaviorTree.Status.Failure)
                    {
                        if (child is LimitNode node)
                        {
                            node.ResetCount();
                        }
                    }
                    return result;
                }
                else
                {
                    return BehaviorTree.Status.Failure;
                }
            }
        }        
        public class MoveBack : BehaviorNode
        {
            public MoveBack (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                if (myRobot.hitByRamming == true)
                {
                    myRobot.hitByRamming = false;
                }
                // If hit by enemy target, set bot facing towards target and run back 
                myRobot.SetTurnRightRadians(myEnemy.GetBearingRadians() + (Math.PI / 2));
                myRobot.SetBack(100);
                return BehaviorTree.Status.Success;
            }
        }
        public class MoveForward : BehaviorNode
        {
            public MoveForward (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {              
                myRobot.SetAhead(250);
                return BehaviorTree.Status.Success;
            }
        }
        public class GetClearShot : BehaviorNode
        {
            public GetClearShot(BFG9000 bot, EnemyBot enemy) : base(bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                myRobot.SetTurnRightRadians(myEnemy.GetBearingRadians() + (Math.PI / 3));
                myRobot.SetAhead(50);
                return BehaviorTree.Status.Success;
            }
        }
        public class MoveTowardsEnemy: BehaviorNode
        {
            public MoveTowardsEnemy(BFG9000 bot, EnemyBot enemy) : base(bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                myRobot.SetTurnRightRadians(myEnemy.GetBearingRadians() + (Math.PI / 2));
                myRobot.SetAhead((myEnemy.GetDistance() / 3));
                return BehaviorTree.Status.Success;
            }
        }
        public class MoveRobot : BehaviorNode
        {
            public MoveRobot (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }

            public override BehaviorTree.Status Process()
            {
                // Stop and Go strategy implementation, detect incremental change in enemy energy change to determin fire shot
                // Direction to determined by current orientation times 36 (enough to move one bot length in a direction)
                // if no change in energy, do not move
                // Reference
                // https://robowiki.net/wiki/Stop_And_Go
                // https://robowiki.net/wiki/Stop_And_Go_Tutorial
                if (myRobot.DistanceRemaining == 0.0)
                {
                    myRobot.SetAhead(myRobot.moveDirection * 36 * Math.Max(0, Math.Sign(myEnemy.GetPrevEnergy() - myEnemy.GetEnergy())));
                }

                return BehaviorTree.Status.Success;                
            }
        }
        public class WideScan : BehaviorNode
        {
            public WideScan (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) {}

            public override BehaviorTree.Status Process()
            {
                myRobot.UpdateIsAdjustGunForRobotTurn(false);
                myRobot.UpdateIsAdjustRadarForGunTurn(false);

                myRobot.SetTurnRadarRightRadians(Double.PositiveInfinity);

                return BehaviorTree.Status.Success;
            }
        }
        public class TargetLockRadar : BehaviorNode
        {
            public TargetLockRadar (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {

                myRobot.UpdateIsAdjustGunForRobotTurn(false);
                myRobot.UpdateIsAdjustRadarForGunTurn(true);

                // Reference Radar tutorial from robowiki
                // https://robowiki.net/wiki/Radar
                // determine angle of turn by adding enemy bearings and my headings reduce by radar heading to get lock on enemy target with radar
                // turn radar independently from gun for faster turn to target
                double radarTurn = Utils.NormalRelativeAngle(myRobot.HeadingRadians + myEnemy.GetBearingRadians() - myRobot.RadarHeadingRadians);
                myRobot.SetTurnRadarRightRadians(radarTurn);
                return BehaviorTree.Status.Success;
            }
        }
        public class NoTargets : ConditionalNode
        {
            public NoTargets (BFG9000 bot, EnemyBot enemy, BehaviorNode child) : base (bot, enemy, child) { }
            protected override bool CheckCondition()
            {
                return myEnemy.NoTarget();
            }
        }
        public class HaveTarget : ConditionalNode
        {
            public HaveTarget(BFG9000 bot, EnemyBot enemy) : base(bot, enemy) { }
            protected override bool CheckCondition()
            {
                return myEnemy.TargetAcquired();
            }
        }
        public class FriendlyInPath : ConditionalNode
        {
            public FriendlyInPath(BFG9000 bot, EnemyBot enemy, BotManager allbots) : base(bot, enemy, allbots) { }
            protected override bool CheckCondition()
            {
                foreach (var friend in myBotManager.GetFriendlies())
                {
                    // check each friendly's coordinates and determine angle between us
                    // calculate gun heading with angle of our distance and see if it falls within 
                    double myX = myRobot.X;
                    double myY = myRobot.Y;
                    double gunHeading = myRobot.GunHeadingRadians;
                    double friendX = friend.x;
                    double friendY = friend.y;
                    double angle = Utils.NormalAbsoluteAngle(Math.Atan2(friendX - myX, friendY - myY));

                    // level of accuracy between angle and gun heading and friendly is close in distance.
                    if((Math.Abs(gunHeading - angle) < 0.3) && (GetDistance(myX, myY, friendX, friendY) < 150))
                    {
                        return true;
                    }
                }
                return false;
            }
        }
        public class AimRadar : BehaviorNode
        {
            public AimRadar (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }

            public override BehaviorTree.Status Process()
            {

                // Reference
                // https://mark.random-article.com/robocode/basic_scanning.html#oscillating
                // Find turn amount to have radar face enemy, my substracting the heading of bot and radar from enemy bearing
                // enemy bot location
                // oscillate the radar by shifting by 30 degrees approx. adjusting the direction each time during game frames
                double radarTurn = myRobot.HeadingRadians - myRobot.RadarHeadingRadians + myEnemy.GetBearingRadians();
                radarTurn += (Math.PI/6) * myRobot.scanDirection;
                myRobot.SetTurnRadarRightRadians(radarTurn);
                myRobot.scanDirection *= -1;

                return BehaviorTree.Status.Success;
            }
        }        
        public class FireGun : BehaviorNode
        {         
            public FireGun (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }            
            public override BehaviorTree.Status Process()
            {
                // calculate enemy distance and use the min between max bullet power
                // and 375 over enemy distance to fire the most conservative bullet if the 
                // enemy is close.
                // Reference 
                // https://mark.random-article.com/robocode/basic_targeting.html
                double distance = myEnemy.GetDistance();
                double power = Math.Min((375 / distance), 3);

                myRobot.SetFire(power);
                return BehaviorTree.Status.Success;
            }
        }          
        public class AimGun : BehaviorNode
        {
            public AimGun (BFG9000 bot, EnemyBot enemy) : base (bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                //Referenced robowiki for circular targeting
                // https://robowiki.net/wiki/Circular_Targeting
                // Get Position of my Robot and field dimensions
                double myX = myRobot.X;
                double myY = myRobot.Y;
                double battleFieldHeight = myRobot.BattleFieldHeight;
                double battleFieldWidth = myRobot.BattleFieldWidth;

                // Find angle of my bot to enemy bot for absolut bearings
                // Determine enemy's coordinates by using their distance from my bots x and y
                // retrieve enemy x and y coordinates from enemy bot class
                double absoluteBearing = myRobot.HeadingRadians + myEnemy.GetBearingRadians();
                double enemyX = myEnemy.GetX();
                double enemyY = myEnemy.GetY();
                double predictedX = enemyX;
                double predictedY = enemyY;

                // Get Enemy Bot heading and previous heading to determine range of change
                // get enemy velocity 
                double enemyHeading = myEnemy.GetHeadingRadians();
                double enemyHeadingChange = enemyHeading - myEnemy.GetPrevHeadingRadians();
                double enemyVelocity = myEnemy.GetVelocity();

                // a counter for tracking time with predictions
                double timer = 0;

                // Bullet power determined by grabbing the min from current robot eneryg out of 3
                // used for calculating speed of bullet in prediction loop
                // https://robowiki.net/wiki/Selecting_Fire_Power
                double bulletPower = Math.Min(3.0, myRobot.Energy);
                double bulletVelocity = 20.0 - (3.0 * bulletPower);

                // prediction loop that determines speed of bullet before it hits predicted location of enemy bot
                // update predicted coordinates of enemy bot by using heading of enemy by its speed
                // break out of loop if predicted coordinates are outside field bounds and update coordinates to adjust to within limits
                // https://robowiki.net/wiki/Circular_Targeting/Walkthrough
                // https://robowiki.net/wiki/Linear_Targeting
                while ((++timer) * bulletVelocity < GetDistance(myX, myY, predictedX, predictedY))
                {

                    //estimate enemy bot's x and y coordinates by using their velocity with their heading to make a prediction
                    // Use sine for calculating x movement or north/south movement by multiplying heading with velocity
                    // use cosine for calculating y movement or east/west movement by multiplying heading with velocity
                    // predict enemy turning rate by updating enemy heading with change in heading
                    double xMovement = Math.Sin(enemyHeading) * enemyVelocity;
                    double yMovement = Math.Cos(enemyHeading) * enemyVelocity;

                    // Update predicted positions with projected x and y coordinates
                    predictedX += xMovement;
                    predictedY += yMovement;
                    enemyHeading += enemyHeadingChange;

                    // Boundaries setup to break out of loop if preditions go outside the field limits, 18 units being the pixel size of tank
                    // Update x and y coordinates to use for angle calculation from bot.
                    if(CheckOutOfBounds(predictedX, predictedY, battleFieldWidth, battleFieldHeight, 18.0))
                    {
                        
                        // Check that predicted x falls with boundary at left bound of 18 and right boundary for field width minus 18
                        // Check that predicted y falls with boundary at top bound of 18 and bottom boundary of field heigh minus 18
                        // Update values of predictions for x and y coordinates of enemy bot
                        predictedX = Math.Min(Math.Max(18.0, predictedX), battleFieldWidth - 18.0);
                        predictedY = Math.Min(Math.Max(18.0, predictedY), battleFieldHeight - 18.0);
                        break;
                    }
                }

                // calculate angle of my bot to predicted position of enemy bot by using the arctan between the points
                // normalize angle to stay within bounds
                double angle = Utils.NormalAbsoluteAngle(Math.Atan2(predictedX - myRobot.X, predictedY - myRobot.Y));

                // turn radar and gun to predicted position of  enemy bot
                myRobot.SetTurnRadarRightRadians(Utils.NormalRelativeAngle(absoluteBearing - myRobot.RadarHeadingRadians));
                myRobot.SetTurnGunRightRadians(Utils.NormalRelativeAngle(angle - myRobot.GunHeadingRadians));

                return BehaviorTree.Status.Success;
            }
        }
        public class EnemyOutRange : ConditionalNode
        {
            public EnemyOutRange (BFG9000 bot, EnemyBot enemy, BehaviorNode childNode) : base (bot, enemy, childNode) { }
            protected override bool CheckCondition()
            {
                return myEnemy.GetDistance() > 600;
            }
        }
        public class RammedByEnemy : ConditionalNode
        {
            public RammedByEnemy (BFG9000 bot, EnemyBot enemy,BehaviorNode child) : base(bot, enemy, child) { }
            protected override bool CheckCondition()
            {
                return myRobot.hitByRamming;               
            }
        }
        public class IsNearWall : ConditionalNode
        {
            public IsNearWall (BFG9000 bot, EnemyBot enemy, BehaviorNode childNode) : base (bot, enemy, childNode) { }

            protected override bool CheckCondition()
            {
                // 36 is about 2 bot's length, setting about a bot's length away from wall
                double boundary = 36;

                // checks if bot is near x plane and checks for y plane walls
                // conditional continues to next step in tree to move tank away from wall
                bool xPlaneWall = myRobot.X < boundary || myRobot.X > myRobot.BattleFieldWidth - boundary; 
                bool yPlaneWall = myRobot.Y < boundary || myRobot.Y > myRobot.BattleFieldHeight - boundary;

                if (xPlaneWall || yPlaneWall)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
        public class AvoidWall : BehaviorNode
        {
            public AvoidWall(BFG9000 bot, EnemyBot enemy) : base(bot, enemy) { }
            public override BehaviorTree.Status Process()
            {
                // Toggle bool for hit wall event 
                myRobot.hitWall = false;

                // Move forward and turn a about 30 degrees to get away from the wall
                myRobot.SetAhead(150); 
                myRobot.SetTurnLeftRadians(Math.PI/6); 
                return BehaviorTree.Status.Success;
            }
        }
        public class AdjustTank : BehaviorNode
        {
            public AdjustTank(BFG9000 bot, EnemyBot enemy) : base(bot,enemy) { }
            public override BehaviorTree.Status Process()
            {
                // Allow free move of gun and radar during turns, adjust tank towards enemy bearings and face towards them to prepare for aiming
                myRobot.UpdateIsAdjustGunForRobotTurn(true);
                myRobot.UpdateIsAdjustRadarForGunTurn(true);

                myRobot.SetTurnRightRadians(myEnemy.GetBearingRadians() + (Math.PI/2));

                return BehaviorTree.Status.Success;
            }
        }
    }
}
