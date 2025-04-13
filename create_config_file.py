def config_file(R, runid, seed, beta):
    currline = """
start

# Constant Parameters

R %d

runid %d

# Seed Values
seed %d
# temperature value
beta %d
end
 """ % (R, runid, seed, beta)
    return currline