//! Golden-value capture harness for the characterization test suite.
//! Run: cargo run --example golden_capture
//! Prints observed outputs of the CURRENT code; values are pasted into tests/.
//! This is a test-only helper and is not part of the library.

use gilgamesh::hw_forward::{
    hw_forward_batch, hw_forward_batch_with_thresholds, hw_forward_single, theta_from_r_bottom,
};
use gilgamesh::layers::linear::{quantize_input, quantize_weights, Linear};
use gilgamesh::network::Network;
use gilgamesh::neurons::lif::PhysicsParams;
use gilgamesh::neurons::{Leaky, NeuronMode};
use gilgamesh::surrogate::SurrogateGradient;
use gilgamesh::training::{AdamOptimizer, NoiseParams, Trainer};
use gilgamesh::config::TrainingConfig;
use gilgamesh::network::NetworkGradients;
use gilgamesh::data::{ArrayDataset, InputEncoder};
use ndarray::{array, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn pf32(label: &str, a: &Array2<f32>) {
    println!("{label} = {:?}", a.iter().cloned().collect::<Vec<f32>>());
}

fn main() {
    println!("================ HW_FORWARD ================");
    // zero input
    let fc1z = Array2::<f32>::zeros((1, 9));
    let fc2z: Vec<Vec<i8>> = (0..9).map(|_| vec![0i8; 10]).collect();
    let z = hw_forward_batch(&fc1z, &fc2z, 1.16, 25);
    pf32("hw_zero", &z);

    // nonzero golden, fixed fc1 and fc2
    let fc1: Array2<f32> = array![[0.5, 1.0, 0.2, 0.8, 0.0, 0.3, 0.9, 0.1, 0.6]];
    let fc2: Vec<Vec<i8>> = vec![
        vec![1, -2, 3, 0, 4, -1, 2, 5, -3, 1],
        vec![-1, 2, 0, 3, -2, 1, 4, -3, 2, 0],
        vec![2, 0, -1, 4, 1, -2, 3, 0, 5, -1],
        vec![0, 3, 2, -1, 4, 0, -2, 1, 3, 2],
        vec![1, -1, 4, 0, 2, 3, -3, 5, 0, 1],
        vec![3, 0, 1, 2, -2, 4, 0, -1, 2, 3],
        vec![-2, 4, 0, 1, 3, -1, 2, 0, 4, -3],
        vec![0, 1, 3, -2, 0, 2, 5, 1, -1, 4],
        vec![2, -3, 1, 4, 0, 1, -2, 3, 2, 0],
    ];
    let nz = hw_forward_batch(&fc1, &fc2, 1.16, 25);
    pf32("hw_nonzero", &nz);

    // batch of 2
    let fc1b: Array2<f32> = array![
        [0.5, 1.0, 0.2, 0.8, 0.0, 0.3, 0.9, 0.1, 0.6],
        [0.1, 0.2, 0.9, 0.3, 0.7, 0.0, 0.4, 0.8, 0.5]
    ];
    let nzb = hw_forward_batch(&fc1b, &fc2, 1.16, 25);
    pf32("hw_nonzero_batch2", &nzb);

    // per-layer thresholds
    let t220 = theta_from_r_bottom(220e3);
    let t150 = theta_from_r_bottom(150e3);
    let nzt = hw_forward_batch_with_thresholds(&fc1, &fc2, 1.16, 25, t220, t150);
    pf32("hw_thresholds", &nzt);

    // single wrapper parity
    let single = hw_forward_single(&[0.5, 1.0, 0.2, 0.8, 0.0, 0.3, 0.9, 0.1, 0.6], &fc2, 1.16, 25);
    println!("hw_single = {single:?}");

    // dac extremes
    let fc1_big: Array2<f32> = Array2::from_elem((1, 9), 1000.0);
    let big = hw_forward_batch(&fc1_big, &fc2, 1.16, 25);
    pf32("hw_big", &big);
    let fc1_neg: Array2<f32> = Array2::from_elem((1, 9), -5.0);
    let neg = hw_forward_batch(&fc1_neg, &fc2, 1.16, 25);
    pf32("hw_neg", &neg);

    println!("theta220 = {:?}", t220);
    println!("theta150 = {:?}", t150);

    println!("================ SURROGATE ================");
    let xs = [-1.0f32, -0.1, 0.0, 0.1, 1.0];
    for v in xs {
        println!(
            "fastsig({v}) = {:?}",
            SurrogateGradient::fast_sigmoid(25.0).backward(v)
        );
        println!("atan({v}) = {:?}", SurrogateGradient::atan(2.0).backward(v));
        println!(
            "sigmoid({v}) = {:?}",
            SurrogateGradient::sigmoid(25.0).backward(v)
        );
        println!(
            "tri({v}) = {:?}",
            SurrogateGradient::Triangular { threshold: 0.5 }.backward(v)
        );
    }
    println!(
        "serde_fastsig = {}",
        serde_json::to_string(&SurrogateGradient::fast_sigmoid(25.0)).unwrap()
    );

    println!("================ LIF ================");
    // Simple mode spike+reset
    let simple = Leaky::new_simple(2, 0.9);
    let st = simple.init_state(1);
    let inp = array![[2.0f32, 0.5]];
    let (sp1, st1, _) = simple.forward(&inp, &st);
    pf32("simple_spikes1", &sp1);
    pf32("simple_mem1", &st1.mem);
    let (sp2, st2, _) = simple.forward(&inp, &st1);
    pf32("simple_spikes2", &sp2);
    pf32("simple_mem2", &st2.mem);

    // Physics default no-spike
    let phys = Leaky::new(3, 0.9);
    let ps = phys.init_state(1);
    let pin = array![[2.0f32, 0.5, 0.3]];
    let (psp, pst, _) = phys.forward(&pin, &ps);
    pf32("phys_default_spikes", &psp);
    pf32("phys_default_mem", &pst.mem);

    // compute_membrane physics vs simple via forward_with_dt
    let physn = Leaky::new_physics(1, 0.01, 0.001);
    let zs = physn.init_state(1);
    let in1 = array![[3.0f32]];
    let dt = 0.001f32;
    let (_, m_dt, _) = physn.forward_with_dt(&in1, &zs, dt);
    pf32("phys_dt", &m_dt.mem);
    let (_, m_2dt, _) = physn.forward_with_dt(&in1, &zs, 2.0 * dt);
    pf32("phys_2dt", &m_2dt.mem);
    let (_, m_hdt, _) = physn.forward_with_dt(&in1, &zs, 0.5 * dt);
    pf32("phys_halfdt", &m_hdt.mem);

    // forward_with_pulse
    let pp = Leaky::new_physics_with_pulse(1, 0.01, 0.001, 0.00167, 4.42);
    let pps = pp.init_state_with_pulse(1);
    let strong = array![[100.0f32]];
    let (pulse1, pps1, _) = pp.forward_with_pulse(&strong, &pps, dt);
    pf32("pulse1", &pulse1);
    let zero = array![[0.0f32]];
    let (pulse2, pps2, _) = pp.forward_with_pulse(&zero, &pps1, dt);
    pf32("pulse2", &pulse2);
    let (pulse3, _, _) = pp.forward_with_pulse(&zero, &pps2, dt);
    pf32("pulse3", &pulse3);

    // forward_with_adaptation
    let adapt =
        Leaky::new_physics_with_threshold_adaptation(2, 0.01, 0.001, 0.001, 1.0, 1.5);
    let mut ast = adapt.init_state_with_adaptation(1);
    let astrong = array![[100.0f32, 100.0]];
    let (asp, ast1, _) = adapt.forward_with_adaptation(&astrong, &ast, dt);
    pf32("adapt_spike1", &asp);
    pf32("adapt_thresh1", ast1.adaptive_threshold.as_ref().unwrap());
    let azero = array![[0.0f32, 0.0]];
    ast = ast1;
    let mut spike_total = 0.0f32;
    spike_total += asp.sum();
    for _ in 0..50 {
        let (s, ns, _) = adapt.forward_with_adaptation(&azero, &ast, dt);
        spike_total += s.sum();
        ast = ns;
    }
    pf32("adapt_thresh_final", ast.adaptive_threshold.as_ref().unwrap());
    println!("adapt_spike_total = {spike_total}");

    // forward_with_hardware_timing
    let hw_mode: NeuronMode = PhysicsParams {
        tau_m: 0.01,
        dt: 0.001,
        comparator_delay_s: 0.0025,
        reset_hold_s: 0.003,
        ..Default::default()
    }
    .into();
    let hwn = Leaky::new(1, 0.9).with_mode(hw_mode).with_threshold(1.0);
    let mut hwst = hwn.init_state_with_hardware_timing(1);
    let hwin = array![[100.0f32]];
    let mut emitted = Vec::new();
    for _ in 0..10 {
        let (out, ns, _) = hwn.forward_with_hardware_timing(&hwin, &hwst, dt);
        emitted.push(out[[0, 0]]);
        hwst = ns;
    }
    println!("hw_timing_emitted = {emitted:?}");
    println!(
        "hw_timing_final_pending = {:?}",
        hwst.pending_spike_steps.as_ref().unwrap().iter().cloned().collect::<Vec<u16>>()
    );
    println!(
        "hw_timing_final_hold = {:?}",
        hwst.reset_hold_steps.as_ref().unwrap().iter().cloned().collect::<Vec<u16>>()
    );

    // forward_full_physics
    let full = Leaky::new_physics_with_pulse(1, 0.01, 0.001, 0.00167, 4.42);
    let mut fst = full.init_state_full(1);
    let fin = array![[100.0f32]];
    let mut full_pulses = Vec::new();
    let mut full_mem = Vec::new();
    let mut full_thresh = Vec::new();
    for _ in 0..5 {
        let (p, ns, _) = full.forward_full_physics(&fin, &fst, dt);
        full_pulses.push(p[[0, 0]]);
        full_mem.push(ns.mem[[0, 0]]);
        full_thresh.push(ns.adaptive_threshold.as_ref().unwrap()[[0, 0]]);
        fst = ns;
    }
    println!("full_pulses = {full_pulses:?}");
    println!("full_mem = {full_mem:?}");
    println!("full_thresh = {full_thresh:?}");

    // forward_noisy_with_dt
    let noisy = Leaky::new_simple(2, 0.9);
    let nst = noisy.init_state(1);
    let nin = array![[1.0f32, 0.8]];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let (nsp, nstate, _) = noisy.forward_noisy_with_dt(&nin, &nst, 0.02, 0.01, &mut rng, 0.001);
    pf32("noisy_spikes", &nsp);
    pf32("noisy_mem", &nstate.mem);

    // backward
    let bw = Leaky::new_simple(2, 0.9);
    let bst = bw.init_state(1);
    let bin = array![[2.0f32, 0.5]];
    let (_, _, cache) = bw.forward(&bin, &bst);
    let gsp = array![[1.0f32, 1.0]];
    let gmn = Array2::<f32>::zeros((1, 2));
    let (gi, gmp) = bw.backward(&gsp, &gmn, &cache);
    pf32("backward_grad_input", &gi);
    pf32("backward_grad_mem_prev", &gmp);

    println!("================ LINEAR ================");
    let lin = Linear::with_seed(3, 2, true, 42);
    pf32("lin_weight", &lin.weight);
    println!("lin_bias = {:?}", lin.bias.as_ref().unwrap().iter().cloned().collect::<Vec<f32>>());
    let lin_in = array![[1.0f32, 2.0, 3.0]];
    pf32("lin_forward", &lin.forward(&lin_in));

    let qw = quantize_weights(&array![[0.1f32, 0.5, -0.3], [0.8, -0.2, 0.4]], 3);
    pf32("quant_weights", &qw);
    let qi = quantize_input(&array![[0.0f32, 0.33, 0.5, 1.0]], 3);
    pf32("quant_input", &qi);

    // backward gradients with synapse gains
    let ling = Linear::with_seed(3, 2, false, 42)
        .with_synapse_gains(1.1, 0.9)
        .with_current_gain(2.0);
    let gin = array![[0.5f32, -0.3, 0.8]];
    let gout = array![[1.0f32, -0.5]];
    let (gi2, gw2, _) = ling.backward(&gin, &gout);
    pf32("ling_grad_input", &gi2);
    pf32("ling_grad_weight", &gw2);

    // forward_noisy seeded
    let linn = Linear::with_seed(3, 2, true, 42);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(7);
    pf32("lin_noisy", &linn.forward_noisy(&lin_in, 0.05, &mut rng2));

    println!("================ NETWORK ================");
    let net = Network::new(49, 100, 10, 0.9, 42);
    let zin = Array2::<f32>::zeros((4, 49));
    let (sc, fm, _) = net.forward(&zin, 25);
    pf32("net_zero_spikes", &sc);
    println!("net_zero_mem_sum = {:?}", fm.sum());

    let cin = Array2::from_elem((4, 49), 0.1f32);
    let (sc2, fm2, _) = net.forward(&cin, 25);
    pf32("net_const_spikes", &sc2);
    pf32("net_const_mem", &fm2);

    let (q8, _, _) = net.forward_quantized(&cin, 10, Some(8));
    pf32("net_q8", &q8);
    let (q3, _, _) = net.forward_quantized(&cin, 10, Some(3));
    pf32("net_q3", &q3);
    let (qf, _, _) = net.forward_quantized_full(&cin, 10, Some(3), true, 4);
    pf32("net_qfull", &qf);

    // backward BPTT
    let (_, _, caches5) = net.forward(&cin, 5);
    let go = Array2::from_elem((4, 10), 0.1f32);
    let grads = net.backward(&cin, &caches5, &go);
    println!("net_grad_fc1_sum = {:?}", grads.fc1_weight.sum());
    println!("net_grad_fc2_sum = {:?}", grads.fc2_weight.sum());
    println!("net_grad_fc1_00 = {:?}", grads.fc1_weight[[0, 0]]);
    println!("net_grad_fc2_00 = {:?}", grads.fc2_weight[[0, 0]]);
    let grads_t = net.backward_truncated(&cin, &caches5, &go, Some(2));
    println!("net_grad_trunc_fc1_sum = {:?}", grads_t.fc1_weight.sum());
    println!("net_grad_trunc_fc2_sum = {:?}", grads_t.fc2_weight.sum());

    // pulse network
    let pnet = Network::new_physics_with_pulse(49, 100, 10, 0.00949, 0.001, 0.00167, 4.42, 42);
    let (psc, _, _) = pnet.forward_with_pulse(&cin, 25, 0.001);
    pf32("net_pulse_spikes", &psc);

    // two-phase pulse
    let mut sp_net = Network::new(49, 100, 10, 0.9, 42);
    sp_net.spike_scale = 0.5;
    let st0 = sp_net.init_state(4);
    let (combined, _, _, _) = sp_net.forward_step(&cin, &st0);
    pf32("net_twophase_step", &combined);

    // variants parity
    let (an0, _, _) = net.forward_with_analog(&cin, 25, 0.0);
    pf32("net_analog0", &an0);
    let enc = InputEncoder::rate_coded(7);
    let (encf, _, _) = net.forward_with_encoding(&cin, &enc, 25);
    pf32("net_encoding", &encf);
    let anet = Network::new_physics_with_adaptation(49, 100, 10, 0.00949, 0.001, 0.001, 1.0, 1.2, 42);
    let (asc, _, _) = anet.forward_with_adaptation(&cin, 25, 0.001);
    pf32("net_adaptation", &asc);
    let (an1, _, _) = net.forward_with_analog(&cin, 25, 0.1);
    pf32("net_analog1", &an1);

    // spiking input
    let mut spk_net = Network::new(49, 100, 10, 0.9, 42);
    spk_net.spiking_input = true;
    let (spk_sc, _, _) = spk_net.forward_quantized(&cin, 5, None);
    pf32("net_spiking_input", &spk_sc);

    println!("================ TRAINING ================");
    let tnet = Network::new(4, 3, 2, 0.9, 42);
    let mut opt = AdamOptimizer::new(&tnet, 1e-3);
    let mut tnet_mut = tnet.clone();
    let mut g = NetworkGradients::zeros_like(&tnet);
    g.fc1_weight.fill(0.01);
    g.fc2_weight.fill(0.01);
    opt.step(&mut tnet_mut, &g);
    println!("adam_step1_fc1_00 = {:?}", tnet_mut.fc1.weight[[0, 0]]);
    println!("adam_step1_fc2_00 = {:?}", tnet_mut.fc2.weight[[0, 0]]);
    opt.step(&mut tnet_mut, &g);
    println!("adam_step2_fc1_00 = {:?}", tnet_mut.fc1.weight[[0, 0]]);
    println!("adam_step2_fc2_00 = {:?}", tnet_mut.fc2.weight[[0, 0]]);
    println!("adam_timestep = {:?}", opt.timestep);

    // Trainer on ArrayDataset replicating MockDataset
    let train_images = Array2::from_shape_fn((32, 4), |(r, c)| ((r + c) as f32).sin() * 0.1);
    let test_images = Array2::from_shape_fn((16, 4), |(r, c)| ((r + c) as f32).cos() * 0.1);
    let train_labels: Vec<usize> = (0..32).map(|i| i % 2).collect();
    let test_labels: Vec<usize> = (0..16).map(|i| i % 2).collect();
    let ds = ArrayDataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
    };
    let net2 = Network::new(4, 3, 2, 0.9, 42);
    let config = TrainingConfig {
        lr: 1e-3,
        epochs: 1,
        batch_size: 8,
        num_steps: 3,
        seed: 42,
        num_workers: 0,
        bptt_steps: None,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
    };
    let mut trainer = Trainer::new(net2, config);
    trainer.noise = NoiseParams::default();
    let (loss, train_acc) = trainer.train_epoch(&ds);
    let test_acc = trainer.evaluate(&ds);
    println!("trainer_loss = {loss:?}");
    println!("trainer_train_acc = {train_acc:?}");
    println!("trainer_test_acc = {test_acc:?}");

    println!("================ CHECKPOINT QUANT ================");
    use gilgamesh::checkpoint::Checkpoint;
    let cnet = Network::new(36, 100, 10, 0.9, 42);
    let cp = Checkpoint::from_network_quantized(&cnet, None, Some(3), Some((6, 6)));
    let q = cp.quantized.as_ref().unwrap();
    println!("cp_magbits = {:?}", q.magnitude_bits);
    println!("cp_maxmag = {:?}", q.max_magnitude);
    println!("cp_fc1_pos = {:?}", q.fc1_pos_scale);
    println!("cp_fc1_neg = {:?}", q.fc1_neg_scale);
    println!("cp_fc1_row0_first5 = {:?}", &q.fc1_weight[0][0..5]);
    println!("cp_img = {:?} {:?}", cp.architecture.image_width, cp.architecture.image_height);

    // emulator fixture
    if std::path::Path::new("models/tarski_36_9_10_emulator.json").exists() {
        let ecp = Checkpoint::load("models/tarski_36_9_10_emulator.json").unwrap();
        let enet = ecp.to_network().unwrap();
        println!("emu_fc1_shape = {:?}", enet.fc1.weight.shape());
        println!("emu_fc2_shape = {:?}", enet.fc2.weight.shape());
        println!("emu_threshold = {:?}", enet.lif1.threshold);
        println!("emu_tau_pulse = {:?}", enet.lif1.mode.tau_pulse());
        let input_vec: Vec<f32> = vec![
            -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, -0.424, 0.246, 0.912, 0.415,
            -0.424, -0.424, -0.250, 0.744, 0.580, 0.912, -0.424, -0.424, -0.424, -0.424, 0.415,
            0.580, -0.424, -0.424, -0.424, 0.080, 0.746, 0.246, -0.424, -0.424, -0.424, 0.415,
            0.580, -0.250, -0.424,
        ];
        let input = Array2::from_shape_vec((1, 36), input_vec).unwrap();
        let (esc, _, _) = enet.forward(&input, 25);
        pf32("emu_spikes", &esc);
    } else {
        println!("emu fixture absent");
    }
}
