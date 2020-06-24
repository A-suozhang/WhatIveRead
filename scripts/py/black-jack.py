import numpy as np
import matplotlib.pyplot as plt
import copy

# define the poker set (using 1 set)
suits = ["diamond", "club", "spade", "heart"]
poker_set = {
    "diamond": np.arange(1,13),
    "club": np.arange(1,13),
    "spade": np.arange(1,13),
    "heart": np.arange(1,13)
}

def draw_one_card(cur_set, verbose=False):
    success = False
    while not success:
        tmp = np.random.randint(52)
        suit = tmp // 13
        num = tmp % 13
        if num in cur_set[suits[suit]]:
            cur_set[suits[suit]] = cur_set[suits[suit]][cur_set[suits[suit]] != num]
            success = True
            if verbose:
                print("Drawed {} of {}".format(num, suits[suit]))
            return num
        else:
            pass


def get_score(hand, trans_ace=False):
    score = 0
    for num in hand:
        if num<=10:
            score += num
        else:
            score += 10
    if not 1 in hand:
        pass
    else:
        # if has ace, could act as 1/11
        score = [score, score+10]
    if trans_ace:
        if isinstance(score,list):
            if max(score)>21:
                score=min(score)
            else:
                score=max(score)
        else:
            pass

    return score



# "Hit" - Add another card
# "Stand" - Show hand with current hands
# "Surrender" - give up with half the chips back
# "Double Down" - if smaller than 15, give a chance to double and add
def action(hand,banker=False):
    if banker:
        # here only consider Ace as 1, cause the smaller will be considered
        # FIXME: check the rule
        banker_below_17 = get_score(hand)<17 if 1 not in hand else get_score(hand)[0]<17
        # try:
        #     banker_below_17 = get_score(hand)<17 if 1 not in hand else get_score(hand)[1]<17
        # except TypeError:
        #     import ipdb; ipdb.set_trace()
        while(banker_below_17):
            hand.append(draw_one_card(cur_set,verbose=False))
            print(hand)
            banker_below_17 = get_score(hand)<17 if 1 not in hand else get_score(hand)[1]<17
        return hand
    else:
        finish=False
        status = 0
        while not finish:
            act = input()
            if act == "hit":
                hand.append(draw_one_card(cur_set,verbose=True))
                print("Hit, add 1 card, Cur hand [{}] score ({})".format(hand,get_score(hand)))
                if get_score(hand,trans_ace=True)>21:
                    finish=True
                    print("Burst!!")
                else:
                    finish=False
            elif act == "surrender":
                finish=True
                print("Surrender, Lost this hand")
                status = -1 # Surrender, Lose
            elif act == "stand":
                finish=True
                print("Stand with current hand")
            elif act == "double down":
                print("Double and add 1 card")
                hand.append(draw_one_card(cur_set,verbose=True))
                print("Double, add 1 card, Cur hand [{}] score ({})".format(hand,get_score(hand)))
                finish=False
                status=2 # Double
        return hand, status

def showhand(banker,player):
    banker_score = get_score(banker,trans_ace=True)
    player_score = get_score(player,trans_ace=True)
    # Check Burst
    if banker_score>21 and player_score>21:
        result = 0
    elif banker_score>21:
        result = 1
    elif player_score>21:
        result = -1
    # check black jack
    else:
        if 21-banker_score<21-player_score:
            result = -1
        elif 21-banker_score>21-player_score:
            result = 1
        else:
            if 21-banker_score == 21-player_score == 0:
                # Check for black jack
                if isinstance(banker_score,list):
                    if isinstance(player_score,list):
                        print("Double Black Jack!")
                    else:
                        result = -1
                    print("Black Jack!")
                elif isinstance(player_score,list):
                    result = 1
                    print("Black Jack")
                else:
                    result = 0
            else:
                result = 0
    print("----- Final Show hand --------")
    print("---Banker {} Score: {}".format(banker,banker_score))
    print("---Player {} Score: {}".format(player,player_score))

    return result

def calc_chips(chipin, result, status):
    if status == -1:
        # Surrenderm Lose half
        return -0.5*chipin
    else:
        if result == 0:
            return 0
        elif result == 1:
            if status == 2:
                return 2*chipin
            else:
                return chipin
        elif result == -1:
            if status == 2:
                return -2*chipin
            else:
                return -chipin

def test_banker():
    banker = []
    global cur_set
    cur_set = copy.deepcopy(poker_set)
    banker.append(draw_one_card(cur_set,verbose=True))
    banker.append(draw_one_card(cur_set)) 
    banker = action(banker,banker=True)
    result = get_score(banker,trans_ace=True)
    return result
    
def main_game(chipin):
# Init 
    banker = []
    player = []
    global cur_set
    cur_set = copy.deepcopy(poker_set)

# 1st: Banker draw 2 card and anounce 1
    print("========= Banker Action =========")
    banker.append(draw_one_card(cur_set,verbose=True))
    banker.append(draw_one_card(cur_set))
    print("=================================")


# 2nd: The Player draw 2 card and choose action
    print("========= Player Action =========")
    player.append(draw_one_card(cur_set,verbose=True))
    player.append(draw_one_card(cur_set,verbose=True))
    print("=== Cur Score {} ===".format(get_score(player)))
    print("========= Please Choose Action =========")

    player, status = action(player)
    banker = action(banker,banker=True)

# showhand([12,10],[1,4,9])
    result = showhand(banker,player)

    results = ["Lose","Draw","Win"]
    print("-------- Final Result :{} ------".format(results[result+1]))

    chipout = calc_chips(chipin, result, status)

    return chipout

if __name__=="__main__":
    # Initializing Player

    init_chips = 100
    cur_chips = init_chips
    print("-- Init Player: Chips {} --".format(cur_chips))
    print("----- Game Start -------")
    while cur_chips>0: 
        print("Please input the chips in for this turn...")
        chipin = int(input())
        if 0<=chipin<=cur_chips:
            pass
        else:
            raise Exception("Ilegal chip num")
        chip_change = main_game(chipin)
        cur_chips += chip_change
        print("-- After this round, chip change {}, current chips {}".format(chip_change, cur_chips))

    print("--- Game Over ---")


    # bankers = []
    # mc_num = 10000
    # for i in range(mc_num):
    #     bankers.append(test_banker())
    # bankers = [-1 if i>21 else i for i in bankers]

    # # P above N is
    # bankers = np.array(bankers)
    # for N in [18,19,20]:
    #     print("The Prob of P>N({}) is:{}".format(N,(bankers>N).sum()/mc_num))
    # 
    # print("The Burst Prob:{}".format((bankers<0).sum()/mc_num))






