import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    eng.cd(r'T1D_VPP/', nargout=0)
    times = [8, 50, 121, 303, 397, 404, 433, 566, 645, 703, 871, 914, 985]
    carbs = [35, 79, 117, 40,  15,  100, 30,  100, 100, 100, 35,  79,  117]

    m_times = matlab.double(times)
    m_carbs = matlab.double(carbs)

    bg, insulin = eng.run_sim(m_times, m_carbs, nargout=2)
    bg, insulin = bg[0], insulin[0]
    t = [i + 1 for i in range(len(bg))]
    t = np.array(t) * (5 / 24 / 60)

    fig, ax = plt.subplots()
    ax.plot(t, bg)

    ax.set(xlabel='time', ylabel='bg',
           title='BG over Time')
    plt.show()
    print(bg)