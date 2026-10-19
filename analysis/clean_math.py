with open('paper_package/13_author_manuscript_draft_iem.md', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    ('$N_g$', 'N_g'),
    (r'$\Delta t = 60\text{ s}$', 'Δt = 60 s'),
    ('$t$', 't'),
    (r'$\mathbf{a}_t^{\text{cloud}} \in \mathcal{A}^{N_g}$', 'a_t^cloud ∈ A^{N_g}'),
    (r'$\tau_{\text{watchdog}} = 120\text{ s}$', 'τ_watchdog = 120 s'),
    (r'$\tau_{\text{watchdog}}$', 'τ_watchdog'),
    (r'$\pi_{\text{local}}$', 'π_local'),
    (r'$g \in \{1, \dots, N_g\}$', 'g ∈ {1, ..., N_g}'),
    (r'$P_g^0(t)$', 'P_g^0(t)'),
    (r'$\alpha_g^{\max}$', 'α_g^max'),
    (r'$\beta_g$', 'β_g'),
    (r'$P_g^{\text{curt}}(t)$', 'P_g^curt(t)'),
    (r'$a_g \in \mathcal{A} = \{0, 0.5, 1.0\}$', 'a_g ∈ {0, 0.5, 1.0}'),
    (r'$P_g^{\text{rebound}}(t)$', 'P_g^rebound(t)'),
    (r'$\tau_{\text{dwell}} = 180\text{ s}$', 'τ_dwell = 180 s'),
    (r'$\boldsymbol{\phi}_{\text{concat}}(s, a) = [s^\top, a^\top]^\top$', 'φ_concat(s, a) = [s^T, a^T]^T'),
    (r'$w_{\text{track}} = 1.0$', 'w_track = 1.0'),
    (r'$w_{\text{curt}} = 0.1$', 'w_curt = 0.1'),
    (r'$w_{\text{switch}} = 0.05$', 'w_switch = 0.05'),
    (r'$T(t)$', 'T(t)'),
    (r'P_g(t,a) = P_g^0(t) - a_g \cdot \alpha_g^{\max} P_g^0(t) + P_g^{\text{rebound}}(t)    (1)',
     'P_g(t, a) = P_g^0(t) - a_g · α_g^max · P_g^0(t) + P_g^rebound(t)    (1)'),
    (r'Q(s, a) = \mathbf{w}^\top \boldsymbol{\phi}(s, a)    (2)',
     'Q(s, a) = w^T · φ(s, a)    (2)'),
    (r'Q(s, a) = \mathbf{w}_a^\top \boldsymbol{\phi}_s    (3)',
     'Q(s, a) = w_a^T · φ_s    (3)'),
    (r'r_t = - \left[ w_{\text{track}} \left(\frac{\max(0, P_{\text{net}}(t) - T(t))}{\Delta P_{\text{flex}}}\right)^2 + w_{\text{curt}} \left(\frac{\sum_g P_g^{\text{curt}}(t)}{\sum_g P_g^0(t)}\right) + w_{\text{switch}} N_{\text{switches}}(t) \right]    (4)',
     'r_t = - [ w_track · (max(0, P_net(t) - T(t)) / ΔP_flex)^2 + w_curt · (Σ_g P_g^curt(t) / Σ_g P_g^0(t)) + w_switch · N_switches(t) ]    (4)'),
    (r'\text{score} = \text{achievement\_rate} - 0.1 \cdot \text{curtail\_fraction} - 0.001 \cdot \text{switches\_per\_run}    (5)',
     'score = achievement_rate - 0.1 · curtail_fraction - 0.001 · switches_per_run    (5)'),
]

cleaned = text
for src, dst in replacements:
    cleaned = cleaned.replace(src, dst)

with open('paper_package/13_author_manuscript_draft_iem.md', 'w', encoding='utf-8') as f:
    f.write(cleaned)

print('Successfully cleaned math replacements in iem.md!')
