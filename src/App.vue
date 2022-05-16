<template>
    <v-app>
        <v-app-bar
            app
            color="primary"
            dark
        >
            <div class="d-flex align-center">
                <h2>Backward propagation</h2>
            </div>
        </v-app-bar>

        <v-main>
            <v-container>
                <v-row>
                    <v-col md="3">
                        <v-text-field
                            v-for="(weight,i) in weights"
                            :key="i"
                            :label="'w'+(i+1)"
                            filled
                            v-model="weights[i]"
                        />
                    </v-col>
                    <v-col md="3">
                        <v-text-field label="bias (1)" filled v-model="bias[0]"/>
                        <v-text-field label="bias (2)" filled v-model="bias[1]"/>

                        <v-text-field label="input (1)" filled v-model="inputs[0]"/>
                        <v-text-field label="input (2)" filled v-model="inputs[1]"/>


                        <v-text-field label="y_value (1)" filled v-model="y_values[0]"/>
                        <v-text-field label="y_value (2)" filled v-model="y_values[1]"/>
                        <v-btn elevation="2" block color="primary" @click="calculate">
                            Hesapla
                        </v-btn>
                    </v-col>
                    <v-col md="6">
                        <div v-for="(output, i) in outputs" :key="i" style="margin-bottom: 30px">
                            <v-card elevation="2">
                                <v-card-title>{{ output.title }}</v-card-title>
                                <v-simple-table>
                                    <template v-slot:default>
                                        <tbody>
                                        <tr v-for="(row, j) in output.rows" :key="j">
                                            <td v-for="(column, k) in row" :key="k">
                                                {{ column }}
                                            </td>
                                        </tr>
                                        </tbody>
                                    </template>
                                </v-simple-table>
                            </v-card>
                        </div>
                        <v-btn elevation="2" block color="primary" @click="next" v-if="nextValues.inputs">
                            Sonraki hesaplama
                        </v-btn>
                    </v-col>
                </v-row>
            </v-container>
        </v-main>
    </v-app>
</template>

<script>

export default {
    name: 'App',
    components: {},
    data: () => ({
        weights: [0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4],
        bias: [0.5, 0.6],
        inputs: [0.04, 0.08],
        y_values: [0.15, 0.85], //Ulaşmaya çalıştığımız gerçek değerler,
        outputs: [],
        nextValues: {}
    }),
    methods: {
        sigmoid(x) {
            return 1 / (1 + Math.exp(-x));
        },
        loss(y_hat_1, y1, y_hat_2, y2) {
            return 1 / 2 * ((y_hat_1 - y1) ** 2 + (y_hat_2 - y2) ** 2)
        },
        calculate() {
            this.outputs = [];
            this.forward_propagation(this.inputs, this.weights, this.bias, this.y_values);
        },
        forward_propagation(inputs, weights, bias, y_values) {
            //liste şeklindeki ağırlıkları kodda kolay anlaşılması açısından
            //temsil ettikleri değişken adlarına atayalım:
            let [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
            let [b1, b2] = bias;
            let [i1, i2] = inputs;
            let [y1, y2] = y_values;

            //ilk hidden layer'ın ve sigmoid'e girdikten sonraki halinin bulunması
            let h1 = i1 * w1 + i2 * w3 + b1
            let h1_out = this.sigmoid(h1)

            //ikinci hidden layer'ın ve sigmoid'e girdikten sonraki halinin bulunması
            let h2 = i1 * w2 + i2 * w4 + b1
            let h2_out = this.sigmoid(h2)

            //ilk output layer'ın ve sigmoid'e girdikten sonraki halinin bulunması
            let o1 = h1 * w5 + h2 * w7 + b2
            let o1_out = this.sigmoid(o1)

            //İkinci output layer'ın ve sigmoid'e girdikten sonraki halinin bulunması
            let o2 = h1 * w6 + h2 * w8 + b2
            let o2_out = this.sigmoid(o2)

            //Bulunan değerlerin ekrana bastırılması (Makaledeki değerler ile de karşılaştırılabilir.)

            this.outputs.push({
                title: 'Forward Propagation',
                rows: [
                    ["h1", h1, "h1_out", h1_out.toFixed(5)],
                    ["h2", h2, "h2_out", h2_out.toFixed(5)],
                    ["o1", o1, "o1_out", o1_out.toFixed(5)],
                    ["o2", o2, "o2_out", o2_out.toFixed(5)],
                    ["Kayıp", this.loss(o1_out, y1, o2_out, y2).toFixed(5)]
                ]
            });

            //Katmanlarımızı bir listeye atıyoruz ki input olarak verirken uzun uzun girmeyelim.
            let hidden_layers = [h1_out, h2_out]
            let output_layers = [o1_out, o2_out]

            //Biraz sonra tanımlayacağımız geriye yayılım fonksiyonunu çağırıyoruz. Böylelikle
            // iç içe fonksiyon yapısı oluşturarak ileri-geri yayılım işlemini beraber yapıyoruz.
            this.backward_propagation(inputs, weights, bias, y_values, hidden_layers, output_layers)
        },
        //Geri yönlü besleme fonksiyonu
        backward_propagation(inputs, weights, bias, y_values, hidden_layers, output_layers) {
            //liste şeklindeki ağırlıkları kodda kolay anlaşılması açısından
            // temsil ettikleri değişken adlarına atayalım:
            let [w1, w2, w3, w4, w5, w6, w7, w8] = weights;
            //let [b1, b2] = bias;
            let [i1, i2] = inputs;
            // eslint-disable-next-line no-unused-vars
            let [y1, y2] = y_values;

            //İleri yönlü beslemede hesapladığımız saklı (hidden) katmanlar
            let [h1_out, h2_out] = hidden_layers;

            //İleri yönlü beslemede hesapladığımız çıkış (output) katmanlar
            let [o1_out, o2_out] = output_layers

            let lr = 0.5 //Makaledeki örneğimizde belirlediğimiz learning rate değer

            // w5'in yeni ağırlığını hesaplamak için makalede hesaplanan türevin kodda yazılması
            let new_w5 = -(y1 - o1_out) * o1_out * (1 - o1_out) * h1_out

            //Yeni ağırlık değeri için türevden gelen değerin learning rate ile çarpılması
            // ve çıkan sonucun eski ağırlıktan çıkarılması
            new_w5 = w5 - lr * new_w5

            //w5'te yukarıda yapıtğımız işlemlerin diğer
            let new_w7 = -(y1 - o1_out) * o1_out * (1 - o1_out) * h2_out
            new_w7 = w7 - lr * new_w7

            let new_w6 = -(y1 - o2_out) * o2_out * (1 - o2_out) * h1_out
            new_w6 = w6 - lr * new_w6

            let new_w8 = -(y1 - o2_out) * o2_out * (1 - o2_out) * h2_out
            new_w8 = w8 - lr * new_w8

            let new_w1 = -(y1 - h1_out) * h1_out * (1 - h1_out) * i1
            new_w1 = w1 - lr * new_w1

            let new_w3 = -(y1 - h1_out) * h1_out * (1 - h1_out) * i2
            new_w3 = w3 - lr * new_w3

            let new_w2 = -(y1 - h2_out) * h2_out * (1 - h2_out) * i1
            new_w2 = w2 - lr * new_w2

            let new_w4 = -(y1 - h2_out) * h2_out * (1 - h2_out) * i2
            new_w4 = w4 - lr * new_w4

            //Geriye yayılımdan hesaplanan değerlerin ekrana bastırılması
            this.outputs.push({
                title: 'Bakward Propagation',
                rows: [
                    ["old w1", w1, "new w1", new_w1.toFixed(5)],
                    ["old w2", w2, "new w2", new_w2.toFixed(5)],
                    ["old w3", w3, "new w3", new_w3.toFixed(5)],
                    ["old w4", w4, "new w4", new_w4.toFixed(5)],
                    ["old w5", w5, "new w5", new_w5.toFixed(5)],
                    ["old w6", w6, "new w6", new_w6.toFixed(5)],
                    ["old w7", w7, "new w7", new_w7.toFixed(5)],
                    ["old w8", w8, "new w8", new_w8.toFixed(5)],
                ]
            })

            //Yeni hesaplanan weight'lerin bir listeye atılması
            weights = [new_w1, new_w2, new_w3, new_w4, new_w5, new_w6, new_w7, new_w8];

            this.nextValues = {inputs, weights, bias, y_values};
        },
        next() {
            this.forward_propagation(
                this.nextValues.inputs,
                this.nextValues.weights,
                this.nextValues.bias,
                this.nextValues.y_values,
            );
        }
    }
};
</script>
