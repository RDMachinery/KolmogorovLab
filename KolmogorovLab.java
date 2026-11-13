/*
 * KolmogorovLab.java
 *
 * A single-file Java Swing laboratory in which a population of tiny programs
 * evolves to draw a target picture in as few bits as possible.
 *
 * The idea behind it is Kolmogorov complexity (Andrey Kolmogorov, Ray Solomonoff,
 * Gregory Chaitin, 1960s): the true information content of a picture is the
 * length of the shortest program that prints it. Nobody can compute that in
 * general, but evolution can search for short programs. Each program here is
 * scored by the Minimum Description Length principle (Jorma Rissanen, 1978):
 * the bits needed to write the program down, plus the bits needed to list the
 * pixels it gets wrong. That total is the size of a real compressed file, which
 * the lab writes, reads back and checks pixel for pixel, and compares with zlib.
 *
 * The search is genetic programming (John Koza, 1992): crossover and mutation of
 * expression trees, with cumulative selection keeping every improvement.
 *
 * Run:   java KolmogorovLab.java        (Java 17 or newer)
 *
 * Architect: Mario Gianota
 * Implementation: Claude (Anthropic)
 * Licence: MIT
 */

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.math.BigInteger;
import java.nio.file.Files;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.stream.IntStream;
import java.util.zip.Deflater;

public class KolmogorovLab extends JFrame {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
            catch (Exception ignored) { }
            new KolmogorovLab().setVisible(true);
        });
    }

    // =====================================================================
    //  The picture and the virtual machine
    // =====================================================================

    static final int W = 64, H = 64, N = W * H;
    static final int BITS_OP = 5, BITS_CONST = 6;
    static final int MAX_NODES = 60;

    // Instruction set. The program is an expression tree, evaluated once per pixel in
    // reading order; the lowest bit of the result is the pixel (1 = ink).
    static final String[] NAME = {"x", "y", "c", "left", "up", "upLeft", "upRight",
            "~", "neg", "abs", "sqr", "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "<", "==", "min", "max", "if"};
    static final int[] ARITY = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3};
    static final int X = 0, Y = 1, C = 2, L = 3, U = 4, UL = 5, UR = 6, NOT = 7, NEG = 8, ABS = 9, SQR = 10,
            ADD = 11, SUB = 12, MUL = 13, DIV = 14, MOD = 15, AND = 16, OR = 17, XOR = 18, SHL = 19, SHR = 20,
            LT = 21, EQ = 22, MIN = 23, MAX = 24, IF = 25;
    static final int NOPS = NAME.length, NTERM = 7;

    // ---- the correction model. For each pixel, a binary arithmetic coder writes whether the program got
    // it wrong. Its prediction uses one of 32 contexts: the program's own pixel, and whether the four
    // neighbours already decoded (left, up, up-left, up-right) were wrong. Counts adapt as it goes (KT
    // estimator). A perfect program never makes a mistake, so its corrections cost almost nothing; mistakes
    // that cluster along edges are cheap; a blank program leaves a plain neighbour model of the picture.

    static final int PONE = 65536;   // probabilities are 16-bit: p out of 65536

    /** -log2(p/65536): the cost in bits of an event of probability p/65536. */
    static final double[] COST = new double[PONE + 1];
    static { for (int p = 1; p <= PONE; p++) COST[p] = -Math.log(p / (double) PONE) / Math.log(2); }

    static final int NCTX = 32;

    /** log2(N choose k): bits to list k wrong pixels as if they were scattered at random. Used as the
     *  formula island's search score, which rewards drawing more pixels right, so near misses count. */
    static final double[] LOG2C = new double[N + 1];
    static {
        double[] lf = new double[N + 1];
        for (int i = 1; i <= N; i++) lf[i] = lf[i - 1] + Math.log(i) / Math.log(2);
        for (int k = 0; k <= N; k++) LOG2C[k] = lf[N] - lf[k] - lf[N - k];
    }

    static int ctx(byte[] err, int sketch, int i, int x, int y, int w) {
        int c = sketch;
        if (x > 0) c |= err[i - 1] << 1;
        if (y > 0) {
            c |= err[i - w] << 2;
            if (x > 0) c |= err[i - w - 1] << 3;
            if (x < w - 1) c |= err[i - w + 1] << 4;
        }
        return c;
    }

    /** Probability (out of 65536) that the next pixel in this context is wrong. */
    static int prob(int n0, int n1) {
        int p = (int) ((2L * n1 + 1) * PONE / (2L * (n0 + n1) + 2));
        return Math.max(1, Math.min(PONE - 1, p));
    }

    static final class Node {
        final int op, c;
        final Node[] k;
        Node(int op, int c, Node... k) { this.op = op; this.c = c; this.k = k; }
        int size() { int s = 1; for (Node n : k) s += n.size(); return s; }
        int consts() { int s = op == C ? 1 : 0; for (Node n : k) s += n.consts(); return s; }
        Node copy() { Node[] kk = new Node[k.length]; for (int i = 0; i < k.length; i++) kk[i] = k[i].copy(); return new Node(op, c, kk); }
        void postfix(ArrayList<Integer> out) {
            for (Node n : k) n.postfix(out);
            out.add(op);
            if (op == C) out.add(c);
        }
        boolean usesNeighbours() {
            if (op >= L && op <= UR) return true;
            for (Node n : k) if (n.usesNeighbours()) return true;
            return false;
        }
    }

    static int[] compile(Node root) {
        ArrayList<Integer> l = new ArrayList<>();
        root.postfix(l);
        int[] code = new int[l.size()];
        for (int i = 0; i < code.length; i++) code[i] = l.get(i);
        return code;
    }

    /** Runs a program over the whole picture. Fills out[]; returns the number of pixels that differ from target (if given). */
    static void run(int[] code, byte[] out) { run(code, out, W, H); }

    static void run(int[] code, byte[] out, int w, int h) { run(code, out, w, h, null, Double.MAX_VALUE, Integer.MAX_VALUE, null); }

    /**
     * Runs a program over the whole picture, filling out[]. With a target, also returns the cost in bits
     * of the corrections (and the number of wrong pixels in errCount[0]); gives up, returning -1, once
     * that cost passes budget, or once both the wrong and the right pixels pass limit.
     */
    static double run(int[] code, byte[] out, int w, int h, byte[] target, double budget, int limit, int[] errCount) {
        int[] st = new int[MAX_NODES * 2 + 8];
        int errs = 0;
        double cost = 0;
        byte[] err = target == null ? null : new byte[w * h];
        int[] n0 = new int[NCTX], n1 = new int[NCTX];
        for (int y = 0, i = 0; y < h; y++) {
            for (int x = 0; x < w; x++, i++) {
                int sp = 0;
                for (int pc = 0; pc < code.length; pc++) {
                    int op = code[pc];
                    switch (op) {
                        case X -> st[sp++] = x;
                        case Y -> st[sp++] = y;
                        case C -> st[sp++] = code[++pc];
                        case L -> st[sp++] = x > 0 ? out[i - 1] : 0;
                        case U -> st[sp++] = y > 0 ? out[i - w] : 0;
                        case UL -> st[sp++] = x > 0 && y > 0 ? out[i - w - 1] : 0;
                        case UR -> st[sp++] = x < w - 1 && y > 0 ? out[i - w + 1] : 0;
                        case NOT -> st[sp - 1] = ~st[sp - 1];
                        case NEG -> st[sp - 1] = -st[sp - 1];
                        case ABS -> st[sp - 1] = Math.abs(st[sp - 1]);
                        case SQR -> st[sp - 1] = st[sp - 1] * st[sp - 1];
                        case IF -> { sp -= 2; st[sp - 1] = st[sp - 1] != 0 ? st[sp] : st[sp + 1]; }
                        default -> {
                            int b = st[--sp], a = st[sp - 1], r;
                            switch (op) {
                                case ADD -> r = a + b;
                                case SUB -> r = a - b;
                                case MUL -> r = a * b;
                                case DIV -> r = b == 0 ? 0 : a / b;
                                case MOD -> r = b == 0 ? 0 : Math.floorMod(a, b);
                                case AND -> r = a & b;
                                case OR -> r = a | b;
                                case XOR -> r = a ^ b;
                                case SHL -> r = a << (b & 31);
                                case SHR -> r = a >> (b & 31);
                                case LT -> r = a < b ? 1 : 0;
                                case EQ -> r = a == b ? 1 : 0;
                                case MIN -> r = Math.min(a, b);
                                default -> r = Math.max(a, b);
                            }
                            st[sp - 1] = r;
                        }
                    }
                }
                byte v = (byte) (st[0] & 1);
                out[i] = v;
                if (target != null) {
                    int c = ctx(err, v, i, x, y, w);
                    int p = prob(n0[c], n1[c]);
                    if (v != target[i]) { cost += COST[p]; n1[c]++; err[i] = 1; errs++; } else { cost += COST[PONE - p]; n0[c]++; }
                    if (cost > budget || (errs > limit && i + 1 - errs > limit)) return -1;
                }
            }
        }
        if (errCount != null) errCount[0] = errs;
        return cost;
    }

    static String infix(Node n) {
        return switch (ARITY[n.op]) {
            case 0 -> n.op == C ? String.valueOf(n.c) : NAME[n.op];
            case 1 -> switch (n.op) {
                case NOT -> "~" + infix(n.k[0]);
                case NEG -> "-" + infix(n.k[0]);
                default -> NAME[n.op] + "(" + infix(n.k[0]) + ")";
            };
            case 2 -> (n.op == MIN || n.op == MAX)
                    ? NAME[n.op] + "(" + infix(n.k[0]) + ", " + infix(n.k[1]) + ")"
                    : "(" + infix(n.k[0]) + " " + NAME[n.op] + " " + infix(n.k[1]) + ")";
            default -> "(" + infix(n.k[0]) + " ? " + infix(n.k[1]) + " : " + infix(n.k[2]) + ")";
        };
    }

    // =====================================================================
    //  A program and its description length
    // =====================================================================

    static final class Program {
        final Node root;
        final int[] code;
        final int nodes, consts;
        int errors;
        double bits, fit;      // bits: the real file size; fit: the formula island's search score
        Program(Node root) {
            this.root = root;
            code = compile(root);
            nodes = root.size();
            consts = root.consts();
        }
        int programBits() { return nodes * BITS_OP + consts * BITS_CONST; }
        static final int OVERHEAD = 40;    // arithmetic coder flush and byte padding, roughly
        void evaluate(byte[] target, double budgetBits, int limit) {
            int[] ec = new int[1];
            double cost = run(code, new byte[N], W, H, target, budgetBits - programBits() - OVERHEAD, limit, ec);
            errors = ec[0];
            bits = cost < 0 ? 1e9 + nodes : programBits() + cost + OVERHEAD;
            fit = cost < 0 ? 1e9 + nodes : programBits() + LOG2C[errors];
        }
    }

    static final Comparator<Program> BY_BITS = (a, b) -> a.bits != b.bits ? Double.compare(a.bits, b.bits) : a.nodes - b.nodes;
    static final Comparator<Program> BY_FIT = (a, b) -> a.fit != b.fit ? Double.compare(a.fit, b.fit) : a.nodes - b.nodes;

    // =====================================================================
    //  The compressed file: program, then the list of wrong pixels
    // =====================================================================

    static final class BitWriter {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        int cur, n;
        long count;
        void write(int value, int bits) { for (int b = bits - 1; b >= 0; b--) bit((value >> b) & 1); }
        void bit(int v) { cur = (cur << 1) | v; n++; count++; if (n == 8) { out.write(cur); cur = 0; n = 0; } }
        byte[] finish() { if (n > 0) { out.write(cur << (8 - n)); cur = 0; n = 0; } return out.toByteArray(); }
    }

    static final class BitReader {
        final byte[] data;
        long pos;
        BitReader(byte[] d) { data = d; }
        int bit() {
            int byteIdx = (int) (pos >> 3);
            if (byteIdx >= data.length) throw new IllegalArgumentException("the file ends too soon");
            int v = (data[byteIdx] >> (7 - (pos & 7))) & 1;
            pos++;
            return v;
        }
        int read(int bits) { int v = 0; for (int i = 0; i < bits; i++) v = (v << 1) | bit(); return v; }
    }

    static BigInteger binom(int n, int k) {
        if (k < 0 || k > n) return BigInteger.ZERO;
        k = Math.min(k, n - k);
        BigInteger c = BigInteger.ONE;
        for (int i = 1; i <= k; i++) c = c.multiply(BigInteger.valueOf(n - k + i)).divide(BigInteger.valueOf(i));
        return c;
    }

    static void writeTree(Node n, BitWriter w) {
        w.write(n.op, BITS_OP);
        if (n.op == C) w.write(n.c, BITS_CONST);
        for (Node k : n.k) writeTree(k, w);
    }

    static Node readTree(BitReader r, int[] budget) {
        if (--budget[0] < 0) throw new IllegalArgumentException("the program is too long");
        int op = r.read(BITS_OP);
        if (op >= NOPS) throw new IllegalArgumentException("unknown instruction " + op);
        int c = op == C ? r.read(BITS_CONST) : 0;
        Node[] k = new Node[ARITY[op]];
        for (int i = 0; i < k.length; i++) k[i] = readTree(r, budget);
        return new Node(op, c, k);
    }

    /** Binary arithmetic coder (carry-less, 32-bit, in the style of Matt Mahoney's lpaq). */
    static final class ArithEncoder {
        long x1 = 0, x2 = 0xffffffffL;
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        void encode(int bit, int p1) {          // p1 = probability of a 1, out of 65536
            long r = x2 - x1, xmid = x1 + (r >>> 16) * p1 + (((r & 0xffff) * p1) >>> 16);
            if (bit != 0) x2 = xmid; else x1 = xmid + 1;
            while (((x1 ^ x2) & 0xff000000L) == 0) {
                out.write((int) (x2 >>> 24));
                x1 = (x1 << 8) & 0xffffffffL;
                x2 = ((x2 << 8) & 0xffffffffL) | 255;
            }
        }
        byte[] finish() { out.write((int) (x1 >>> 24)); return out.toByteArray(); }   // the decoder pads with 255s
    }

    static final class ArithDecoder {
        long x1 = 0, x2 = 0xffffffffL, x = 0;
        final byte[] in;
        int pos;
        ArithDecoder(byte[] in, int start) { this.in = in; pos = start; for (int i = 0; i < 4; i++) x = (x << 8) | next(); }
        private int next() { return pos < in.length ? in[pos++] & 255 : 255; }
        int decode(int p1) {
            long r = x2 - x1, xmid = x1 + (r >>> 16) * p1 + (((r & 0xffff) * p1) >>> 16);
            int bit = x <= xmid ? 1 : 0;
            if (bit != 0) x2 = xmid; else x1 = xmid + 1;
            while (((x1 ^ x2) & 0xff000000L) == 0) {
                x1 = (x1 << 8) & 0xffffffffL;
                x2 = ((x2 << 8) & 0xffffffffL) | 255;
                x = ((x << 8) & 0xffffffffL) | next();
            }
            return bit;
        }
    }

    /** The exact compressed file: the program tree (a prefix code), padded to a byte, then the corrections, arithmetic-coded. */
    static byte[] encode(Node prog, byte[] target) { return encode(prog, target, W, H); }

    static byte[] encode(Node prog, byte[] target, int w, int h) {
        BitWriter bw = new BitWriter();
        writeTree(prog, bw);
        byte[] head = bw.finish();
        byte[] out = new byte[w * h];
        run(compile(prog), out, w, h);
        ArithEncoder ac = new ArithEncoder();
        byte[] err = new byte[w * h];
        int[] n0 = new int[NCTX], n1 = new int[NCTX];
        for (int y = 0, i = 0; y < h; y++) for (int x = 0; x < w; x++, i++) {
            int e = out[i] ^ target[i], c = ctx(err, out[i], i, x, y, w);
            ac.encode(e, prob(n0[c], n1[c]));
            if (e != 0) { n1[c]++; err[i] = 1; } else n0[c]++;
        }
        byte[] tail = ac.finish();
        byte[] file = Arrays.copyOf(head, head.length + tail.length);
        System.arraycopy(tail, 0, file, head.length, tail.length);
        return file;
    }

    static Node decodedProgram;

    static byte[] decode(byte[] file) { return decode(file, W, H); }

    static byte[] decode(byte[] file, int w, int h) {
        BitReader r = new BitReader(file);
        Node prog = readTree(r, new int[]{10_000});
        decodedProgram = prog;
        byte[] sketch = new byte[w * h], img = new byte[w * h], err = new byte[w * h];
        run(compile(prog), sketch, w, h);
        ArithDecoder ad = new ArithDecoder(file, (int) ((r.pos + 7) / 8));
        int[] n0 = new int[NCTX], n1 = new int[NCTX];
        for (int y = 0, i = 0; y < h; y++) for (int x = 0; x < w; x++, i++) {
            int c = ctx(err, sketch[i], i, x, y, w);
            int e = ad.decode(prob(n0[c], n1[c]));
            if (e != 0) { n1[c]++; err[i] = 1; } else n0[c]++;
            img[i] = (byte) (sketch[i] ^ e);
        }
        return img;
    }

    static byte[] pack(byte[] img) {
        byte[] p = new byte[(img.length + 7) / 8];
        for (int i = 0; i < img.length; i++) if (img[i] != 0) p[i >> 3] |= (byte) (0x80 >> (i & 7));
        return p;
    }

    /** zlib's deflate at maximum effort, raw stream (no header), on the packed bitmap. */
    static int zlibBytes(byte[] img) {
        Deflater d = new Deflater(9, true);
        d.setInput(pack(img));
        d.finish();
        byte[] buf = new byte[4096];
        int n = 0;
        while (!d.finished()) n += d.deflate(buf);
        d.end();
        return n;
    }

    // =====================================================================
    //  Evolution
    // =====================================================================

    static final class Engine {
        byte[] target = new byte[N];
        int zlib, coderAlone;
        Program[] pop;                          // both islands together, best first
        Program[][] islands = new Program[2][];
        static final boolean[] NEIGHBOURS = {false, true};   // island 0: formulas only; island 1: may read drawn pixels
        Random rng;
        int popSize = 500;
        long gen, evals;
        private boolean nb;                     // the island being bred

        Engine(long seed, byte[] t) { setBaselines(t); reset(seed); }

        private void setBaselines(byte[] t) {
            target = t.clone();
            zlib = zlibBytes(target);
            coderAlone = encode(new Node(C, 0), target).length;   // the empty program: the corrections coder does everything
        }

        void reset(long seed) {
            rng = new Random(seed);
            for (int s = 0; s < 2; s++) {
                nb = NEIGHBOURS[s];
                Program[] p = new Program[popSize / 2];
                for (int i = 0; i < p.length; i++) p[i] = new Program(randomTree(1 + i % 5, i % 2 == 0));
                islands[s] = p;
                evaluate(s, p, 0, Double.MAX_VALUE, N);
            }
            merge();
            gen = 0;
        }

        void setTarget(byte[] t) {
            setBaselines(t);
            for (int s = 0; s < 2; s++) evaluate(s, islands[s], 0, Double.MAX_VALUE, N);
            merge();
        }

        private void evaluate(int island, Program[] p, int from, double budget, int limit) {
            byte[] t = target;
            IntStream.range(from, p.length).parallel().forEach(i -> p[i].evaluate(t, budget, limit));
            evals += p.length - from;
            Arrays.sort(p, island == 0 ? BY_FIT : BY_BITS);
        }

        /** Wrong pixels beyond which a program can't beat a search score of fit, whatever its size. */
        private static int limitFor(double fit) {
            int k = 0;
            while (k < N / 2 && LOG2C[k + 1] <= fit) k++;
            return k;
        }

        private void merge() {
            pop = new Program[islands[0].length + islands[1].length];
            System.arraycopy(islands[0], 0, pop, 0, islands[0].length);
            System.arraycopy(islands[1], 0, pop, islands[0].length, islands[1].length);
            Arrays.sort(pop, BY_BITS);
        }

        Program best() { return pop[0]; }

        private Program[] cur;

        void step() {
            for (int s = 0; s < 2; s++) {
                nb = NEIGHBOURS[s];
                cur = islands[s];
                int size = cur.length;
                Program[] next = new Program[size];
                int elite = Math.min(4, size);
                System.arraycopy(cur, 0, next, 0, elite);
                for (int i = elite; i < size; i++) {
                    Node child = breed();
                    if (child.size() > MAX_NODES || (!nb && child.usesNeighbours())) child = tournament().root;
                    next[i] = new Program(child);
                }
                // a program clearly worse than the island's median is abandoned early
                double budget = s == 0 ? Double.MAX_VALUE : cur[size / 2].bits;
                int limit = s == 0 ? limitFor(cur[size / 2].fit) : N;
                islands[s] = next;
                evaluate(s, next, elite, budget, limit);
            }
            gen++;
            if (gen % 25 == 0) {   // migration: each island's best visits the other, if it may live there
                Program a = islands[0][0], b = islands[1][0];
                Program[] i1 = islands[1], i0 = islands[0];
                i1[i1.length - 1] = a;
                Arrays.sort(i1, BY_BITS);
                if (!b.root.usesNeighbours()) { i0[i0.length - 1] = b; Arrays.sort(i0, BY_FIT); }
            }
            merge();
        }

        Program tournament() {
            int best = rng.nextInt(cur.length);
            for (int i = 1; i < 4; i++) best = Math.min(best, rng.nextInt(cur.length));
            return cur[best];
        }

        Node breed() {
            double r = rng.nextDouble();
            Node a = tournament().root;
            if (r < 0.40) return crossover(a, tournament().root);
            if (r < 0.60) return pointMutation(a);
            if (r < 0.75) return replaceAt(a, rng.nextInt(a.size()), randomTree(rng.nextInt(4), false));
            if (r < 0.82) return hoist(a);
            if (r < 0.89) return shrink(a);
            if (r < 0.97) return wrap(a);
            return randomTree(1 + rng.nextInt(5), rng.nextBoolean());
        }

        // ---- tree surgery (all operations return new trees; parents are never changed)

        static void preorder(Node n, ArrayList<Node> out) { out.add(n); for (Node k : n.k) preorder(k, out); }

        static Node nodeAt(Node root, int idx) { ArrayList<Node> l = new ArrayList<>(); preorder(root, l); return l.get(idx); }

        static Node replaceAt(Node root, int idx, Node repl) {
            if (idx == 0) return repl;
            Node copy = root.copy();
            replaceRec(copy, idx, repl, new int[]{0});
            return copy;
        }

        private static boolean replaceRec(Node n, int idx, Node repl, int[] count) {
            for (int i = 0; i < n.k.length; i++) {
                count[0]++;
                if (count[0] == idx) { n.k[i] = repl; return true; }
                if (replaceRec(n.k[i], idx, repl, count)) return true;
            }
            return false;
        }

        Node crossover(Node a, Node b) {
            Node donor = nodeAt(b, rng.nextInt(b.size())).copy();
            return replaceAt(a, rng.nextInt(a.size()), donor);
        }

        Node pointMutation(Node a) {
            int idx = rng.nextInt(a.size());
            Node n = nodeAt(a, idx);
            Node repl;
            if (n.op == C && rng.nextBoolean()) {
                int c = Math.max(0, Math.min(63, n.c + (rng.nextBoolean() ? 1 : -1) * (1 + rng.nextInt(4))));
                repl = new Node(C, c);
            } else if (ARITY[n.op] == 0) repl = randomTerminal();
            else {
                int op;
                do { op = NTERM + rng.nextInt(NOPS - NTERM); } while (ARITY[op] != ARITY[n.op]);
                Node[] k = new Node[n.k.length];
                for (int i = 0; i < k.length; i++) k[i] = n.k[i].copy();
                repl = new Node(op, 0, k);
            }
            return replaceAt(a, idx, repl);
        }

        Node hoist(Node a) { return nodeAt(a, rng.nextInt(a.size())).copy(); }

        Node shrink(Node a) {
            ArrayList<Integer> fn = new ArrayList<>();
            ArrayList<Node> l = new ArrayList<>();
            preorder(a, l);
            for (int i = 0; i < l.size(); i++) if (ARITY[l.get(i).op] > 0) fn.add(i);
            if (fn.isEmpty()) return pointMutation(a);
            return replaceAt(a, fn.get(rng.nextInt(fn.size())), randomTerminal());
        }

        Node wrap(Node a) {
            int idx = rng.nextInt(a.size());
            Node inner = nodeAt(a, idx).copy();
            int op = NTERM + rng.nextInt(NOPS - NTERM);
            Node[] k = new Node[ARITY[op]];
            int slot = rng.nextInt(k.length);
            for (int i = 0; i < k.length; i++) k[i] = i == slot ? inner : randomTree(rng.nextInt(2), false);
            return replaceAt(a, idx, new Node(op, 0, k));
        }

        Node randomTerminal() {
            int op = nb ? rng.nextInt(NTERM) : rng.nextInt(3);
            if (op == C) return new Node(C, rng.nextDouble() < 0.6 ? rng.nextInt(9) : rng.nextInt(64));
            return new Node(op, 0);
        }

        Node randomTree(int depth, boolean full) {
            if (depth <= 0 || (!full && rng.nextDouble() < 0.35)) return randomTerminal();
            int op = NTERM + rng.nextInt(NOPS - NTERM);
            Node[] k = new Node[ARITY[op]];
            for (int i = 0; i < k.length; i++) k[i] = randomTree(depth - 1, full);
            return new Node(op, 0, k);
        }
    }

    // =====================================================================
    //  Target pictures
    // =====================================================================

    static final String[] PRESETS = {"Sierpinski triangle", "Checkerboard", "Circle", "Rings", "Diagonal stripes",
            "Rule 30 automaton", "Smiley face", "Text: Hi!", "Random noise", "Blank (draw your own)"};

    static byte[] preset(String name) {
        byte[] t = new byte[N];
        switch (name) {
            case "Sierpinski triangle" -> { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) t[y * W + x] = (byte) ((x & y) == 0 ? 1 : 0); }
            case "Checkerboard" -> { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) t[y * W + x] = (byte) (((x >> 3) ^ (y >> 3)) & 1); }
            case "Circle" -> { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { int dx = x - 32, dy = y - 32; t[y * W + x] = (byte) (dx * dx + dy * dy < 22 * 22 ? 1 : 0); } }
            case "Rings" -> { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { int dx = x - 32, dy = y - 32; t[y * W + x] = (byte) (((dx * dx + dy * dy) >> 6) & 1); } }
            case "Diagonal stripes" -> { for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) t[y * W + x] = (byte) (((x + y) >> 2) & 1); }
            case "Rule 30 automaton" -> {
                t[32] = 1;
                for (int y = 1; y < H; y++) for (int x = 0; x < W; x++) {
                    int l = x > 0 ? t[(y - 1) * W + x - 1] : 0, c = t[(y - 1) * W + x], r = x < W - 1 ? t[(y - 1) * W + x + 1] : 0;
                    t[y * W + x] = (byte) (l ^ (c | r));
                }
            }
            case "Smiley face" -> {
                BufferedImage img = canvas();
                Graphics2D g = img.createGraphics();
                g.setColor(Color.BLACK);
                g.setStroke(new BasicStroke(3f));
                g.draw(new Ellipse2D.Double(5, 5, 53, 53));
                g.fill(new Ellipse2D.Double(20, 19, 7, 10));
                g.fill(new Ellipse2D.Double(37, 19, 7, 10));
                g.drawArc(17, 22, 30, 24, 200, 140);
                g.dispose();
                return fromImage(img);
            }
            case "Text: Hi!" -> {
                BufferedImage img = canvas();
                Graphics2D g = img.createGraphics();
                g.setColor(Color.BLACK);
                g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 34));
                FontMetrics fm = g.getFontMetrics();
                String s = "Hi!";
                g.drawString(s, (W - fm.stringWidth(s)) / 2, (H + fm.getAscent() - fm.getDescent()) / 2);
                g.dispose();
                return fromImage(img);
            }
            case "Random noise" -> { Random r = new Random(1); for (int i = 0; i < N; i++) t[i] = (byte) r.nextInt(2); }
            default -> { }
        }
        return t;
    }

    static BufferedImage canvas() {
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, W, H);
        g.dispose();
        return img;
    }

    static byte[] fromImage(BufferedImage src) {
        BufferedImage img = canvas();
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        double s = Math.min((double) W / src.getWidth(), (double) H / src.getHeight());
        int w = (int) Math.round(src.getWidth() * s), h = (int) Math.round(src.getHeight() * s);
        g.drawImage(src, (W - w) / 2, (H - h) / 2, w, h, null);
        g.dispose();
        byte[] t = new byte[N];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int rgb = img.getRGB(x, y);
            int lum = (((rgb >> 16) & 255) * 30 + ((rgb >> 8) & 255) * 59 + (rgb & 255) * 11) / 100;
            t[y * W + x] = (byte) (lum < 128 ? 1 : 0);
        }
        return t;
    }

    // =====================================================================
    //  Application state
    // =====================================================================

    private final ConcurrentLinkedQueue<Runnable> tasks = new ConcurrentLinkedQueue<>();
    private volatile boolean running = true, singleStep = false;
    private volatile Engine engine;
    private final byte[] editTarget;                          // the UI's copy, edited by drawing
    private volatile boolean targetDirty = false;

    record Snapshot(long gen, double gensPerSec, long evals, Program best, byte[] bestOut, byte[] target,
                    int zlibBytes, int coderAlone, long fileBits, int fileBytes, boolean verified, String expr,
                    byte[][] thumbs, double[] thumbBits, double[] histGen, double[] histBits) { }

    private volatile Snapshot snap;
    private final ArrayList<double[]> history = new ArrayList<>();
    private Node lastVerified = null;
    private byte[] lastFile = null;
    private long lastFileBits;
    private boolean lastVerifiedOk;

    private final ImagePanel targetPanel = new ImagePanel("Target (draw on it)", true);
    private final ImagePanel outputPanel = new ImagePanel("Best program's drawing", false);
    private final ImagePanel errorPanel = new ImagePanel("Pixels it gets wrong", false);
    private final SizeBars bars = new SizeBars();
    private final HistoryChart chart = new HistoryChart();
    private final PopulationGrid grid = new PopulationGrid();
    private final JTextArea programText = new JTextArea(14, 34);
    private final JLabel status = new JLabel(" ");

    private final JComboBox<String> presetBox = new JComboBox<>(PRESETS);
    private final JTextField seedField = new JTextField("1", 6);
    private final JToggleButton playButton = new JToggleButton("Pause", true);
    private final ParamSlider popSlider = new ParamSlider("Population (programs)", 100, 2000, 500, "%.0f");
    private final JComboBox<String> brushBox = new JComboBox<>(new String[]{"1 pixel", "2 pixels", "3 pixels", "5 pixels"});

    public KolmogorovLab() {
        super("Kolmogorov Lab — evolving the shortest program that draws a picture");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        editTarget = preset(PRESETS[0]);
        engine = new Engine(1, editTarget);
        brushBox.setSelectedIndex(1);
        buildUI();
        Thread worker = new Thread(this::loop, "evolution");
        worker.setDaemon(true);
        worker.start();
        new javax.swing.Timer(100, e -> refresh()).start();
        pack();
        setLocationRelativeTo(null);
    }

    // =====================================================================
    //  Evolution thread
    // =====================================================================

    private void loop() {
        long lastSnap = 0, lastT = System.nanoTime(), lastGen = 0;
        double gps = 0;
        while (true) {
            Runnable t;
            while ((t = tasks.poll()) != null) t.run();
            Engine e = engine;
            if (targetDirty) {
                byte[] copy;
                synchronized (editTarget) { copy = editTarget.clone(); targetDirty = false; }
                e.setTarget(copy);
                history.clear();
                lastVerified = null;
            }
            if (running || singleStep) {
                singleStep = false;
                e.step();
                Program b = e.best();
                if (history.isEmpty() || history.get(history.size() - 1)[1] != b.bits) {
                    history.add(new double[]{e.gen, b.bits});
                    if (history.size() > 3000) {
                        ArrayList<double[]> thin = new ArrayList<>();
                        for (int i = 0; i < history.size(); i += 2) thin.add(history.get(i));
                        history.clear(); history.addAll(thin);
                    }
                }
            } else {
                try { Thread.sleep(20); } catch (InterruptedException ex) { return; }
            }
            long now = System.nanoTime();
            if (now - lastT > 1_000_000_000L) { gps = (e.gen - lastGen) * 1e9 / (now - lastT); lastGen = e.gen; lastT = now; }
            if (now - lastSnap > 150_000_000L || !running) { snapshot(e, gps); lastSnap = now; }
        }
    }

    private void snapshot(Engine e, double gps) {
        Program best = e.best();
        if (best.root != lastVerified) {
            byte[] file = encode(best.root, e.target);
            byte[] back = decode(file);
            lastVerifiedOk = Arrays.equals(back, e.target);
            lastFile = file;
            lastFileBits = file.length * 8L;
            lastVerified = best.root;
        }
        byte[] out = new byte[N];
        run(best.code, out);
        int nt = 24;
        byte[][] thumbs = new byte[nt][];
        double[] tb = new double[nt];
        for (int i = 0; i < nt; i++) {
            int idx = (int) Math.round(Math.pow((double) i / (nt - 1), 2) * (e.pop.length - 1));
            Program p = e.pop[idx];
            thumbs[i] = new byte[N];
            run(p.code, thumbs[i]);
            tb[i] = p.bits;
        }
        double[] hg = new double[history.size() + 1], hb = new double[history.size() + 1];
        for (int i = 0; i < history.size(); i++) { hg[i] = history.get(i)[0]; hb[i] = history.get(i)[1]; }
        hg[hg.length - 1] = e.gen; hb[hb.length - 1] = best.bits;
        snap = new Snapshot(e.gen, gps, e.evals, best, out, e.target.clone(), e.zlib, e.coderAlone, lastFileBits, lastFile.length,
                lastVerifiedOk, infix(best.root), thumbs, tb, hg, hb);
    }

    // =====================================================================
    //  UI refresh
    // =====================================================================

    private void refresh() {
        Snapshot s = snap;
        if (s == null) return;
        byte[] tgt;
        synchronized (editTarget) { tgt = editTarget.clone(); }
        targetPanel.set(tgt, null);
        outputPanel.set(s.bestOut, null);
        errorPanel.set(s.target, s.bestOut);
        bars.repaint();
        chart.repaint();
        grid.repaint();
        Program b = s.best;
        int rawBytes = N / 8;
        StringBuilder sb = new StringBuilder();
        sb.append("pixel(x, y) = lowest bit of\n\n    ").append(s.expr).append("\n\n");
        int ink = 0;
        for (byte v : s.bestOut) ink += v;
        if (ink == 0 || ink == N)
            sb.append("This program draws a blank page, so the corrections coder is doing all the work.\n\n");
        else if (b.root.usesNeighbours())
            sb.append("(left, up, upLeft and upRight are pixels this program has already drawn.)\n\n");
        int progBytes = (b.programBits() + 7) / 8;
        sb.append(String.format(Locale.ROOT, "Program:      %d instructions, %d constants = %d bits (%d bytes)%n", b.nodes, b.consts, b.programBits(), progBytes));
        sb.append(String.format(Locale.ROOT, "Corrections:  %d wrong pixels = %d bytes%n", b.errors, s.fileBytes - progBytes));
        sb.append(String.format(Locale.ROOT, "Total file:   %d bytes%n%n", s.fileBytes));
        sb.append(String.format(Locale.ROOT, "Raw bitmap:   %d bytes   (%.1f× smaller)%n", rawBytes, (double) rawBytes / s.fileBytes));
        if (b.errors > N / 2) sb.append("(Over half the pixels are wrong: it's drawn the\nnegative, and the coder learns to flip them.)\n");
        sb.append(String.format(Locale.ROOT, "zlib -9:      %d bytes   (%s)%n", s.zlibBytes, ratioWords(s.zlibBytes, s.fileBytes)));
        sb.append(String.format(Locale.ROOT, "Coder alone:  %d bytes   (%s)%n%n", s.coderAlone,
                s.fileBytes < s.coderAlone ? String.format(Locale.ROOT, "evolution saves %d bytes", s.coderAlone - s.fileBytes)
                        : "evolution hasn't beaten it yet"));
        sb.append(s.verified ? "Lossless check: passed. The file was decoded and\nmatches the target pixel for pixel."
                : "Lossless check: FAILED.");
        String txt = sb.toString();
        if (!programText.getText().equals(txt)) programText.setText(txt);
        status.setText(String.format(Locale.ROOT,
                "generation %,d   %.0f gen/s   programs tried %,d   best %d bytes   raw %d bytes   zlib %d bytes",
                s.gen, s.gensPerSec, s.evals, s.fileBytes, rawBytes, s.zlibBytes));
    }

    private static String ratioWords(int zlib, int ours) {
        if (ours < zlib) return String.format(Locale.ROOT, "%.1f× smaller", (double) zlib / ours);
        if (ours == zlib) return "a tie";
        return String.format(Locale.ROOT, "zlib wins by %.1f×", (double) ours / zlib);
    }

    // =====================================================================
    //  Actions
    // =====================================================================

    private long readSeed() {
        try { return Long.parseLong(seedField.getText().trim()); }
        catch (NumberFormatException e) { seedField.setText("1"); return 1; }
    }

    private void setEditTarget(byte[] t) {
        synchronized (editTarget) { System.arraycopy(t, 0, editTarget, 0, N); targetDirty = true; }
    }

    private void restart() {
        long seed = readSeed();
        int size = (int) Math.round(popSlider.value());
        tasks.add(() -> {
            byte[] t;
            synchronized (editTarget) { t = editTarget.clone(); targetDirty = false; }
            Engine e = new Engine(seed, t);
            if (size != e.popSize) { e.popSize = size; e.reset(seed); }
            engine = e;
            history.clear();
            lastVerified = null;
        });
    }

    private void loadImage() {
        JFileChooser fc = new JFileChooser();
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            BufferedImage img = ImageIO.read(fc.getSelectedFile());
            if (img == null) throw new Exception("not an image this program can read");
            setEditTarget(fromImage(img));
        } catch (Exception ex) { JOptionPane.showMessageDialog(this, "Could not load the picture: " + ex.getMessage()); }
    }

    private void saveCompressed() {
        Snapshot s = snap;
        byte[] file = lastFile;
        if (s == null || file == null) return;
        JFileChooser fc = new JFileChooser();
        fc.setSelectedFile(new File("picture.kol"));
        if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            Files.write(fc.getSelectedFile().toPath(), file);
            JOptionPane.showMessageDialog(this, String.format(Locale.ROOT,
                    "Saved %d bytes. Open it again with \"Open compressed file\" to see it decoded.", file.length));
        } catch (Exception ex) { JOptionPane.showMessageDialog(this, "Could not save: " + ex.getMessage()); }
    }

    private void openCompressed() {
        JFileChooser fc = new JFileChooser();
        if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
        try {
            byte[] data = Files.readAllBytes(fc.getSelectedFile().toPath());
            byte[] img = decode(data);
            String prog = infix(decodedProgram);
            setEditTarget(img);
            JOptionPane.showMessageDialog(this, String.format(Locale.ROOT,
                    "Decoded a %d-byte file into a %d×%d picture.%n%nIts program: pixel(x, y) = %s%n%n" +
                    "The picture is now the target, and evolution carries on from here.", data.length, W, H,
                    prog.length() > 200 ? prog.substring(0, 200) + "…" : prog));
        } catch (Exception ex) { JOptionPane.showMessageDialog(this, "That isn't a file this lab can decode: " + ex.getMessage()); }
    }

    // ---- Scale up: the same program, drawn with more pixels

    record ScaleRow(int size, byte[] img, int raw, int zlib, int evolved, boolean ok) { }

    private void showScaleUp() {
        Snapshot s = snap;
        if (s == null) return;
        Node prog = s.best.root;
        String expr = s.expr;
        int[] sizes = {64, 128, 256, 512, 1024};
        JDialog d = new JDialog(this, "Scale up: the same program, more pixels", false);
        JLabel picture = new JLabel("Drawing…", SwingConstants.CENTER);
        JScrollPane ps = new JScrollPane(picture);
        ps.setPreferredSize(new Dimension(560, 560));
        ps.getViewport().setBackground(new Color(0x30, 0x30, 0x36));
        JComboBox<String> which = new JComboBox<>();
        for (int sz : sizes) which.addItem(sz + " × " + sz);
        which.setEnabled(false);
        String[] cols = {"Size", "Raw bitmap", "zlib -9", "Evolved", "zlib ÷ evolved", "Lossless"};
        javax.swing.table.DefaultTableModel model = new javax.swing.table.DefaultTableModel(cols, 0) {
            @Override public boolean isCellEditable(int r, int c) { return false; }
        };
        JTable table = new JTable(model);
        table.setRowHeight(22);
        JScrollPane ts = new JScrollPane(table);
        ts.setPreferredSize(new Dimension(520, 140));
        JTextArea note = new JTextArea(6, 40);
        note.setEditable(false);
        note.setLineWrap(true);
        note.setWrapStyleWord(true);
        note.setOpaque(false);
        note.setFont(note.getFont().deriveFont(13f));
        note.setText("pixel(x, y) = " + expr + "\n\nWorking out each size…");

        JPanel right = new JPanel(new BorderLayout(0, 10));
        right.setBorder(new EmptyBorder(10, 10, 10, 10));
        right.add(ts, BorderLayout.NORTH);
        right.add(note, BorderLayout.CENTER);
        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT));
        top.add(new JLabel("Show at: "));
        top.add(which);
        top.add(new JLabel("  (shown at one screen pixel per picture pixel; scroll to see it all)"));
        JPanel leftP = new JPanel(new BorderLayout());
        leftP.setBorder(new EmptyBorder(10, 10, 10, 0));
        leftP.add(top, BorderLayout.NORTH);
        leftP.add(ps, BorderLayout.CENTER);
        d.getContentPane().setLayout(new BorderLayout());
        d.getContentPane().add(leftP, BorderLayout.CENTER);
        d.getContentPane().add(right, BorderLayout.EAST);
        d.pack();
        d.setLocationRelativeTo(this);
        d.setVisible(true);

        ScaleRow[] rows = new ScaleRow[sizes.length];
        new SwingWorker<Void, ScaleRow>() {
            @Override protected Void doInBackground() {
                int[] code = compile(prog);
                for (int k = 0; k < sizes.length; k++) {
                    int n = sizes[k];
                    byte[] img = new byte[n * n];
                    KolmogorovLab.run(code, img, n, n);
                    byte[] file = encode(prog, img, n, n);
                    boolean ok = Arrays.equals(decode(file, n, n), img);
                    rows[k] = new ScaleRow(n, img, n * n / 8, zlibBytes(img), file.length, ok);
                    publish(rows[k]);
                }
                return null;
            }
            @Override protected void process(List<ScaleRow> chunk) {
                for (ScaleRow r : chunk) {
                    model.addRow(new Object[]{r.size() + " × " + r.size(), String.format(Locale.ROOT, "%,d bytes", r.raw()),
                            String.format(Locale.ROOT, "%,d bytes", r.zlib()), String.format(Locale.ROOT, "%,d bytes", r.evolved()),
                            String.format(Locale.ROOT, "%,.1f×", (double) r.zlib() / r.evolved()), r.ok() ? "yes" : "NO"});
                }
            }
            @Override protected void done() {
                which.setEnabled(true);
                which.addActionListener(e -> showScaled(picture, rows[which.getSelectedIndex()]));
                which.setSelectedIndex(sizes.length - 1);
                ScaleRow a = rows[0], b = rows[sizes.length - 1];
                note.setText(String.format(Locale.ROOT,
                        "pixel(x, y) = %s%n%n" +
                        "From %d×%d to %d×%d the picture has %,d times as many pixels. zlib's file grew from %,d to %,d bytes. " +
                        "The evolved file went from %,d to %,d bytes: the program is the same few bytes at every size, and only " +
                        "the corrections coder's small overhead changes.%n%n" +
                        "Evolution saw only the 64×64 target. At larger sizes the picture is simply whatever this program " +
                        "draws: the same rule, with more pixels. Each file was decoded again and checked pixel for pixel.",
                        expr, a.size(), a.size(), b.size(), b.size(), (b.size() * b.size()) / (a.size() * a.size()),
                        a.zlib(), b.zlib(), a.evolved(), b.evolved()));
            }
        }.execute();
    }

    private static void showScaled(JLabel label, ScaleRow r) {
        int n = r.size();
        BufferedImage bi = new BufferedImage(n, n, BufferedImage.TYPE_INT_RGB);
        int paper = PAPER.getRGB(), ink = INK.getRGB();
        int[] px = new int[n * n];
        for (int i = 0; i < px.length; i++) px[i] = r.img()[i] != 0 ? ink : paper;
        bi.setRGB(0, 0, n, n, px, 0, n);
        int scale = n <= 128 ? 4 : n <= 256 ? 2 : 1;
        Image shown = scale == 1 ? bi : bi.getScaledInstance(n * scale, n * scale, Image.SCALE_REPLICATE);
        label.setText(null);
        label.setIcon(new ImageIcon(shown));
    }

    private void showAbout() {
        String html = """
            <html><body style='width:540px; font-family:sans-serif'>
            <h2>Kolmogorov Lab</h2>
            <p>How much information is there in a picture? In the 1960s Andrey Kolmogorov,
            Ray Solomonoff and Gregory Chaitin gave the deepest answer we have: the length of the
            <b>shortest program</b> that draws it. A Sierpinski triangle looks intricate, but
            <tt>(x &amp; y) == 0</tt> draws the whole thing. A picture of random noise has no shorter
            description than itself.</p>
            <p>No program can find the shortest description of everything; it's provably
            uncomputable. But evolution can <i>search</i> for short ones. Here a population of
            tiny programs, each an arithmetic expression worked out once for every pixel, competes to
            draw the target. Programs can also look at pixels they have already drawn, the one to the
            left and the three above, which lets them copy, continue and grow patterns.</p>
            <p>Each program is scored by the <b>Minimum Description Length</b> principle (Jorma
            Rissanen, 1978): the bits needed to write the program down, plus the bits needed to correct
            the pixels it gets wrong. The corrections are written by a small adaptive arithmetic coder
            that learns where mistakes tend to be, from whether the neighbouring pixels were wrong.
            The total is the size of a real compressed file. The lab writes the file, decodes it again,
            checks it matches the target exactly, and compares it with zlib (the compressor inside ZIP
            and PNG) and with the corrections coder working alone, so you can see what evolution
            itself adds.</p>
            <p>The search is <b>genetic programming</b> (John Koza, 1992). Programs are bred by
            swapping pieces between them (crossover) and by small random changes (mutation), and
            every improvement is kept and built on: cumulative selection, in the space of programs.
            The population lives on two <b>islands</b>. On one, programs are pure formulas of x
            and y, rewarded for every pixel they draw correctly, so near misses count. On the other,
            programs may also read the pixels they've already drawn, and are judged purely on file
            size. Every 25 generations the best of each visits the other.</p>
            <p><b>What to expect.</b> Pictures made by a rule are where evolution shines: it can
            find files a few bytes long where zlib needs a hundred. Drawn shapes and text are
            harder: there the corrections coder does most of the work, and evolution's job is to find
            a rough sketch that leaves fewer mistakes to correct. Random noise can't be compressed by
            anything.</p>
            <p><b>Fine print.</b> Like zlib's file, ours leaves out the picture size and assumes
            the reader has the decoder. Ours is the 26-instruction machine described here.</p>
            <p style='color:#666'>Architect: Mario Gianota · Implementation: Claude (Anthropic) · MIT licence</p>
            </body></html>""";
        JOptionPane.showMessageDialog(this, new JLabel(html), "About Kolmogorov Lab", JOptionPane.PLAIN_MESSAGE);
    }

    // =====================================================================
    //  UI construction
    // =====================================================================

    private void buildUI() {
        JPanel left = new JPanel();
        left.setLayout(new BoxLayout(left, BoxLayout.Y_AXIS));
        left.setBorder(new EmptyBorder(8, 8, 8, 10));

        JPanel tgt = section("Target picture");
        presetBox.addActionListener(e -> setEditTarget(preset((String) presetBox.getSelectedItem())));
        tgt.add(row("Picture: ", presetBox));
        JPanel tb = new JPanel(new GridLayout(2, 2, 4, 4));
        JButton clear = new JButton("Clear");
        JButton invert = new JButton("Invert");
        JButton load = new JButton("Load image…");
        clear.addActionListener(e -> setEditTarget(new byte[N]));
        invert.addActionListener(e -> {
            byte[] t;
            synchronized (editTarget) { t = editTarget.clone(); }
            for (int i = 0; i < N; i++) t[i] ^= 1;
            setEditTarget(t);
        });
        load.addActionListener(e -> loadImage());
        tb.add(clear); tb.add(invert); tb.add(load);
        tb.setAlignmentX(LEFT_ALIGNMENT);
        tb.setMaximumSize(new Dimension(300, tb.getPreferredSize().height));
        tgt.add(Box.createVerticalStrut(4));
        tgt.add(tb);
        tgt.add(row("Brush: ", brushBox));
        JLabel hint = new JLabel("<html><small>Draw on the target with the mouse. Right-drag (or Shift) erases.</small></html>");
        hint.setAlignmentX(LEFT_ALIGNMENT);
        tgt.add(hint);
        left.add(tgt);

        JPanel ev = section("Evolution");
        JButton restartBtn = new JButton("Restart");
        restartBtn.setToolTipText("A fresh random population, same target");
        restartBtn.addActionListener(e -> restart());
        JPanel sr = row("Seed: ", seedField);
        sr.add(Box.createHorizontalStrut(6));
        sr.add(restartBtn);
        ev.add(sr);
        popSlider.onRelease(this::restart);
        ev.add(popSlider);
        JPanel eb = new JPanel(new GridLayout(1, 2, 4, 4));
        JButton step = new JButton("Step");
        playButton.addActionListener(e -> { running = playButton.isSelected(); playButton.setText(running ? "Pause" : "Play"); });
        step.addActionListener(e -> singleStep = true);
        eb.add(playButton); eb.add(step);
        eb.setAlignmentX(LEFT_ALIGNMENT);
        eb.setMaximumSize(new Dimension(300, eb.getPreferredSize().height));
        ev.add(Box.createVerticalStrut(4));
        ev.add(eb);
        left.add(ev);

        JPanel fl = section("Files and scale");
        JPanel fb = new JPanel(new GridLayout(4, 1, 4, 4));
        JButton scale = new JButton("Scale up…");
        scale.setToolTipText("Draw the best program at sizes up to 1024 × 1024 and compare file sizes");
        scale.addActionListener(e -> showScaleUp());
        fb.add(scale);
        JButton save = new JButton("Save compressed file…");
        JButton open = new JButton("Open compressed file…");
        JButton about = new JButton("About");
        save.addActionListener(e -> saveCompressed());
        open.addActionListener(e -> openCompressed());
        about.addActionListener(e -> showAbout());
        fb.add(save); fb.add(open); fb.add(about);
        fb.setAlignmentX(LEFT_ALIGNMENT);
        fb.setMaximumSize(new Dimension(300, fb.getPreferredSize().height));
        fl.add(fb);
        left.add(fl);
        left.add(Box.createVerticalGlue());

        // centre: the three pictures, the size bars and the history
        JPanel pics = new JPanel(new GridLayout(1, 3, 8, 0));
        pics.add(targetPanel); pics.add(outputPanel); pics.add(errorPanel);
        JPanel centre = new JPanel(new BorderLayout(0, 8));
        centre.setBorder(new EmptyBorder(8, 4, 8, 4));
        centre.add(pics, BorderLayout.NORTH);
        JPanel lower = new JPanel(new BorderLayout(0, 8));
        bars.setBorder(BorderFactory.createTitledBorder("Size of the picture, four ways"));
        chart.setBorder(BorderFactory.createTitledBorder("Best description so far (bits, log scale)"));
        lower.add(bars, BorderLayout.NORTH);
        lower.add(chart, BorderLayout.CENTER);
        centre.add(lower, BorderLayout.CENTER);

        // right: the program and the population
        JPanel right = new JPanel(new BorderLayout(0, 8));
        right.setBorder(new EmptyBorder(8, 4, 8, 8));
        programText.setEditable(false);
        programText.setLineWrap(true);
        programText.setWrapStyleWord(true);
        programText.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane ps = new JScrollPane(programText);
        ps.setBorder(BorderFactory.createTitledBorder("Best program"));
        right.add(ps, BorderLayout.CENTER);
        grid.setBorder(BorderFactory.createTitledBorder("The population, best to worst"));
        right.add(grid, BorderLayout.SOUTH);
        right.setPreferredSize(new Dimension(350, 700));

        status.setBorder(new EmptyBorder(4, 10, 4, 10));
        status.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));

        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(left, BorderLayout.WEST);
        getContentPane().add(centre, BorderLayout.CENTER);
        getContentPane().add(right, BorderLayout.EAST);
        getContentPane().add(status, BorderLayout.SOUTH);
    }

    private static JPanel row(String label, JComponent c) {
        JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 2));
        p.add(new JLabel(label));
        p.add(c);
        p.setAlignmentX(LEFT_ALIGNMENT);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, p.getPreferredSize().height));
        return p;
    }

    private static JPanel section(String title) {
        JPanel p = new JPanel();
        p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
        p.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), title, TitledBorder.LEFT, TitledBorder.TOP),
                new EmptyBorder(4, 6, 6, 6)));
        p.setAlignmentX(LEFT_ALIGNMENT);
        p.setMaximumSize(new Dimension(300, Integer.MAX_VALUE));
        return p;
    }

    // =====================================================================
    //  Widgets
    // =====================================================================

    static final Color PAPER = new Color(0xf4, 0xef, 0xe4), INK = new Color(0x1d, 0x24, 0x33);

    static final class ParamSlider extends JPanel {
        private final int TICKS;
        private final double min, max;
        private final String fmt;
        private final JSlider slider;
        private final JLabel valueLabel = new JLabel();
        private Runnable release = () -> { };
        ParamSlider(String name, double min, double max, double initial, String fmt) {
            this.min = min; this.max = max; this.fmt = fmt;
            TICKS = (int) Math.round(max - min);
            slider = new JSlider(0, TICKS);
            setLayout(new BorderLayout());
            setAlignmentX(LEFT_ALIGNMENT);
            setMaximumSize(new Dimension(Integer.MAX_VALUE, 52));
            JPanel top = new JPanel(new BorderLayout());
            JLabel nameLabel = new JLabel(name);
            nameLabel.setFont(nameLabel.getFont().deriveFont(12f));
            valueLabel.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            top.add(nameLabel, BorderLayout.WEST);
            top.add(valueLabel, BorderLayout.EAST);
            add(top, BorderLayout.NORTH);
            add(slider, BorderLayout.CENTER);
            slider.addChangeListener(e -> { refresh(); if (!slider.getValueIsAdjusting()) release.run(); });
            slider.setValue((int) Math.round((initial - min) / (max - min) * TICKS));
            refresh();
        }
        double value() { return min + (max - min) * slider.getValue() / TICKS; }
        void onRelease(Runnable r) { release = r; }
        private void refresh() { valueLabel.setText(String.format(Locale.ROOT, fmt, value())); }
    }

    final class ImagePanel extends JPanel {
        private final String title;
        private byte[] img, compare;
        private int lastPaint = -1;
        ImagePanel(String title, boolean editable) {
            this.title = title;
            setPreferredSize(new Dimension(220, 242));
            if (editable) {
                MouseAdapter m = new MouseAdapter() {
                    @Override public void mousePressed(MouseEvent e) { paintAt(e); }
                    @Override public void mouseDragged(MouseEvent e) { paintAt(e); }
                };
                addMouseListener(m);
                addMouseMotionListener(m);
                setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
            }
        }
        void set(byte[] img, byte[] compare) { this.img = img; this.compare = compare; repaint(); }
        Rectangle area() {
            int s = Math.max(1, Math.min(getWidth() - 4, getHeight() - 24) / W);
            int w = s * W;
            return new Rectangle((getWidth() - w) / 2, 20, w, w);
        }
        private void paintAt(MouseEvent e) {
            Rectangle r = area();
            int px = (e.getX() - r.x) * W / r.width, py = (e.getY() - r.y) * H / r.height;
            int brush = new int[]{1, 2, 3, 5}[brushBox.getSelectedIndex()];
            byte v = (byte) (SwingUtilities.isRightMouseButton(e) || e.isShiftDown() ? 0 : 1);
            synchronized (editTarget) {
                for (int dy = 0; dy < brush; dy++) for (int dx = 0; dx < brush; dx++) {
                    int x = px + dx - brush / 2, y = py + dy - brush / 2;
                    if (x >= 0 && x < W && y >= 0 && y < H) editTarget[y * W + x] = v;
                }
                targetDirty = true;
            }
            repaint();
        }
        @Override protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g.setColor(Color.DARK_GRAY);
            g.drawString(title, 4, 14);
            if (img == null) return;
            Rectangle r = area();
            BufferedImage bi = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
            int paper = PAPER.getRGB(), ink = INK.getRGB();
            int red = new Color(225, 50, 50).getRGB(), faint = new Color(0xd8, 0xd2, 0xc4).getRGB();
            for (int i = 0; i < N; i++) {
                int c;
                if (compare == null) c = img[i] != 0 ? ink : paper;
                else c = img[i] != compare[i] ? red : img[i] != 0 ? faint : paper;
                bi.setRGB(i % W, i / W, c);
            }
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            g.drawImage(bi, r.x, r.y, r.width, r.height, null);
            g.setColor(Color.GRAY);
            g.drawRect(r.x - 1, r.y - 1, r.width + 1, r.height + 1);
        }
    }

    final class SizeBars extends JPanel {
        SizeBars() { setPreferredSize(new Dimension(700, 158)); }
        @Override protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Snapshot s = snap;
            if (s == null) return;
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            Insets in = getInsets();
            int x0 = in.left + 165, w = getWidth() - x0 - in.right - 170, y = in.top + 8;
            double raw = N, zl = s.zlibBytes * 8.0, ca = s.coderAlone * 8.0, ours = s.fileBits;
            double max = Math.max(Math.max(raw, ca), Math.max(zl, ours));
            int progBits = s.best.programBits();
            String[] labels = {"Raw bitmap", "zlib (deflate -9)", "Corrections coder alone", "Evolved program"};
            double[] vals = {raw, zl, ca, ours};
            String[] notes = {String.format(Locale.ROOT, "%d bytes", N / 8), String.format(Locale.ROOT, "%d bytes", s.zlibBytes),
                    String.format(Locale.ROOT, "%d bytes", s.coderAlone), String.format(Locale.ROOT, "%d bytes", s.fileBytes)};
            for (int i = 0; i < 4; i++) {
                int bw = (int) Math.max(2, vals[i] / max * w);
                g.setColor(Color.DARK_GRAY);
                g.drawString(labels[i], in.left + 8, y + 15);
                if (i < 3) {
                    g.setColor(new Color(170, 170, 178));
                    g.fillRect(x0, y + 3, bw, 16);
                } else {
                    int pw = (int) Math.max(1, Math.min(bw, progBits / max * w));
                    g.setColor(new Color(40, 110, 200));
                    g.fillRect(x0, y + 3, pw, 16);
                    g.setColor(new Color(230, 110, 90));
                    g.fillRect(x0 + pw, y + 3, Math.max(0, bw - pw), 16);
                }
                g.setColor(Color.DARK_GRAY);
                g.drawString(notes[i], x0 + bw + 8, y + 15);
                y += 28;
            }
            g.setFont(g.getFont().deriveFont(11f));
            g.setColor(new Color(40, 110, 200));
            g.fillRect(x0, y + 2, 10, 10);
            g.setColor(Color.DARK_GRAY);
            g.drawString("the program", x0 + 14, y + 11);
            g.setColor(new Color(230, 110, 90));
            g.fillRect(x0 + 100, y + 2, 10, 10);
            g.setColor(Color.DARK_GRAY);
            g.drawString("the corrections", x0 + 114, y + 11);
        }
    }

    final class HistoryChart extends JPanel {
        HistoryChart() { setPreferredSize(new Dimension(700, 200)); }
        @Override protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Snapshot s = snap;
            if (s == null || s.histGen.length < 1) return;
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            Insets in = getInsets();
            int x0 = in.left + 50, y0 = in.top + 10, w = getWidth() - x0 - in.right - 90, h = getHeight() - y0 - in.bottom - 28;
            if (w < 20 || h < 20) return;
            double lo = Math.log10(16), hi = Math.log10(N * 2.0);
            double gmax = Math.max(10, s.histGen[s.histGen.length - 1]);
            java.util.function.DoubleUnaryOperator Y = v -> y0 + h - (Math.log10(Math.max(16, v)) - lo) / (hi - lo) * h;
            java.util.function.DoubleUnaryOperator Xs = v -> x0 + v / gmax * w;
            g.setFont(g.getFont().deriveFont(11f));
            for (int v : new int[]{16, 64, 256, 1024, 4096}) {
                int yy = (int) Y.applyAsDouble(v);
                g.setColor(new Color(225, 225, 230));
                g.drawLine(x0, yy, x0 + w, yy);
                g.setColor(Color.GRAY);
                String lab = v + "";
                g.drawString(lab, x0 - 6 - g.getFontMetrics().stringWidth(lab), yy + 4);
            }
            // reference lines
            double[] refs = {N, s.zlibBytes * 8.0};
            String[] rl = {"raw", "zlib"};
            for (int i = 0; i < 2; i++) {
                int yy = (int) Y.applyAsDouble(refs[i]);
                g.setColor(new Color(150, 150, 160));
                g.setStroke(new BasicStroke(1f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 1f, new float[]{4, 4}, 0));
                g.drawLine(x0, yy, x0 + w, yy);
                g.setStroke(new BasicStroke(1f));
                g.drawString(rl[i], x0 + w + 6, yy + 4);
            }
            // the best-so-far staircase
            g.setColor(new Color(40, 110, 200));
            g.setStroke(new BasicStroke(2f));
            int px = -1, py = -1;
            for (int i = 0; i < s.histGen.length; i++) {
                int xx = (int) Xs.applyAsDouble(s.histGen[i]), yy = (int) Y.applyAsDouble(s.histBits[i]);
                if (px >= 0) { g.drawLine(px, py, xx, py); g.drawLine(xx, py, xx, yy); }
                px = xx; py = yy;
            }
            g.setStroke(new BasicStroke(1f));
            g.setColor(Color.GRAY);
            g.drawString("generation " + String.format(Locale.ROOT, "%,d", s.gen), x0 + w - 100, y0 + h + 18);
            g.drawString("0", x0, y0 + h + 18);
        }
    }

    final class PopulationGrid extends JPanel {
        PopulationGrid() { setPreferredSize(new Dimension(340, 290)); }
        @Override protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Snapshot s = snap;
            if (s == null) return;
            Graphics2D g = (Graphics2D) g0;
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            Insets in = getInsets();
            int cols = 6, rows = 4;
            int cw = (getWidth() - in.left - in.right - 8) / cols, chh = (getHeight() - in.top - in.bottom - 4) / rows;
            int sz = Math.min(cw - 6, chh - 18);
            g.setFont(g.getFont().deriveFont(10f));
            int paper = PAPER.getRGB(), ink = INK.getRGB();
            for (int i = 0; i < s.thumbs.length && i < cols * rows; i++) {
                int x = in.left + 4 + (i % cols) * cw, y = in.top + 2 + (i / cols) * chh;
                BufferedImage bi = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
                byte[] t = s.thumbs[i];
                for (int p = 0; p < N; p++) bi.setRGB(p % W, p / W, t[p] != 0 ? ink : paper);
                g.drawImage(bi, x, y, sz, sz, null);
                g.setColor(Color.GRAY);
                g.drawRect(x, y, sz, sz);
                g.drawString(s.thumbBits[i] >= 1e9 ? "gave up" : String.format(Locale.ROOT, "%.0f bits", s.thumbBits[i]), x, y + sz + 12);
            }
        }
    }
}
