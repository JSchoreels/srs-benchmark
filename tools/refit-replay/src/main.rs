//! FSRS-7 re-fitted from scratch during the review stream (fsrs_refit.py).
//!
//! usage: refit-replay <in.bin> <out.bin>
//!
//! in.bin (little-endian): i32 n_items, i32 n_reviews, i32 start, i32 every, i64
//! offsets[n_items + 1], f32 delta_t[n_reviews], i32 rating[n_reviews], i64
//! card_id[n_items], i64 review_th[n_items], i64 prev_th[n_items], i32 day[n_items].
//! Items are in review_th order; item k's last review is the one predicted. `start` is the
//! first item to predict. Fits happen before item `start` and then before every `every`-th
//! item after it, or, with every = 0, before the first item of each day. A fit uses all
//! items before it (fsrs::benchmark, FSRS-7).
//!
//! out.bin: f64 p_latest[n_items - start] (newest fit before the item), f64
//! p_sched[n_items - start] (newest fit whose training items all come at or before the
//! card's previous review, prev_th: what a scheduler had), f64 fits, f64 fit seconds.

use std::fs;
use std::time::Instant;

use fsrs::{
    ComputeParametersInput, ComputeParametersVersion, FSRSItem, FSRSReview, InstantFsrs7,
    InstantReview, benchmark,
};

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self, n: usize) -> Vec<[u8; N]> {
        let out = self.bytes[self.pos..self.pos + N * n]
            .chunks_exact(N)
            .map(|c| c.try_into().unwrap())
            .collect();
        self.pos += N * n;
        out
    }
    fn i32s(&mut self, n: usize) -> Vec<i32> {
        self.take::<4>(n).into_iter().map(i32::from_le_bytes).collect()
    }
    fn i64s(&mut self, n: usize) -> Vec<i64> {
        self.take::<8>(n).into_iter().map(i64::from_le_bytes).collect()
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.take::<4>(n).into_iter().map(f32::from_le_bytes).collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).expect("read input");
    let mut rd = Reader { bytes: &bytes, pos: 0 };
    let head = rd.i32s(4);
    let (n_items, n_reviews) = (head[0] as usize, head[1] as usize);
    let (start, every) = (head[2] as usize, head[3] as usize);
    let offsets = rd.i64s(n_items + 1);
    let delta_t = rd.f32s(n_reviews);
    let rating = rd.i32s(n_reviews);
    let card_id = rd.i64s(n_items);
    let review_th = rd.i64s(n_items);
    let prev_th = rd.i64s(n_items);
    let day = rd.i32s(n_items);
    assert_eq!(rd.pos, bytes.len(), "trailing input");

    let span = |k: usize| offsets[k] as usize..offsets[k + 1] as usize;
    let items: Vec<FSRSItem> = (0..n_items)
        .map(|k| FSRSItem {
            reviews: span(k)
                .map(|j| FSRSReview { rating: rating[j] as u32, delta_t: delta_t[j] })
                .collect(),
        })
        .collect();

    // fit points: (first item not in the training set, parameters)
    let mut fits: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut fit_secs = 0.0;
    let mut p_latest = Vec::with_capacity(n_items - start);
    let mut p_sched = Vec::with_capacity(n_items - start);
    for k in start..n_items {
        let due = k == start
            || if every == 0 { day[k] != day[k - 1] } else { (k - start) % every == 0 };
        if due {
            let t0 = Instant::now();
            let params = benchmark(ComputeParametersInput {
                train_set: items[..k].to_vec(),
                card_ids: Some(card_id[..k].to_vec()),
                enable_short_term: true,
                model_version: ComputeParametersVersion::Fsrs7,
                ..Default::default()
            });
            fit_secs += t0.elapsed().as_secs_f64();
            fits.push((k, params));
        }
        let history: Vec<InstantReview> = span(k)
            .take(span(k).len() - 1)
            .map(|j| InstantReview { delta_t: delta_t[j], rating: rating[j] as u32 })
            .collect();
        let t = delta_t[offsets[k + 1] as usize - 1];
        p_latest.push(InstantFsrs7::predict_with(&fits.last().unwrap().1, &history, t));
        // newest fit whose training items (all before its point) end at or before prev_th
        let sched = fits
            .iter()
            .rev()
            .find(|(point, _)| *point == 0 || review_th[*point - 1] <= prev_th[k])
            .unwrap_or(&fits[0]);
        p_sched.push(InstantFsrs7::predict_with(&sched.1, &history, t));
    }
    let mut out: Vec<u8> = Vec::new();
    for p in p_latest.iter().chain(&p_sched) {
        out.extend(p.to_le_bytes());
    }
    out.extend((fits.len() as f64).to_le_bytes());
    out.extend(fit_secs.to_le_bytes());
    fs::write(&args[2], out).expect("write output");
}
