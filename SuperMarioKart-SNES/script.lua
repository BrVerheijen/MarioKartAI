function getLap()
    --133 final
    return data.lap-128
end

function getMilisec()
    return data.currMiliSec - 300
end

function getCheckPoint()
    local checkpoint = data.current_checkpoint
    local lapsize = data.lapsize
    local lap = data.lap-128
    --local rank = data.rank/2+1

    return checkpoint + (lap)*lapsize
end

function isTurnedAround()
    return data.isTurnedAround == 0x10
end

function isDone()
    if data.getGameMode ~= 0x1C then
        return true
    end

    local lap = getLap
    if lap >= 5 then --for now
        return true
    end

    return false
end

data.prevCheckpoint = getCheckPoint
data.prevFrame = data.getFrame

function getCheckPointReward()
    local newCheckpoint = getCheckPoint()

    local curFrame = data.getFrame
    if curFrame < data.prevFrame or curFrame > data.prevFrame + 60 then
        data.prevCheckpoint = getCheckPoint
    end
    data.pre = curFrame

    local reward = 0
    --- this will give rewards of 100 on each cross
    reward = reward + (newCheckpoint - data.prevCheckpoint) * 10
    data.prevCheckpoint = newCheckpoint

    -- Sanity check
    if reward < -5000 then
        return 0
    end

    return reward
end

function getExperimentalReward()

    local reward = 0

    if data.surface == 128 then
        --hit a wall
        reward= -1
    end
    if data.surface == 40 or data.surface == 32 or data.surface== 34 then
        --fell off, or deep dived
        reward=-1
    end

    return reward
end

wall_hits=0
wall_steps=0
function isHittingWall()
    
    wall_steps = wall_steps + 1

    if data.surface == 128 or data.surface == 40 or data.surface == 32 then
        wall_hits = wall_hits + 1
    end

    if data.surface == 34 then
        wall_hits= wall_hits + 0.05
    end

    if wall_hits >= 5 then
        
        wall_hits = 0
        wall_steps = 0

        return true
    end

    if wall_steps == 500 then
        wall_hits = 0
        wall_steps = 0
    end

    return false
end

function getRewardTrain()
    return getCheckPointReward() + getExperimentalReward()
end

function isDoneTrain()
    return isDone() or isHittingWall()
end