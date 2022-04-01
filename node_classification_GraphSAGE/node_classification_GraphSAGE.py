from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from sklearn.manifold import TSNE
from tensorflow.python.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')

def plot_embeddings(data, subjects):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data[:, 0], data[:, 1], c=subjects.astype('category').cat.codes,
               cmap='jet', alpha=0.7)
    ax.set(aspect='equal', xlabel='$X_1$', ylabel='$X_2$',
           title=f'TSNE visualization of GCN embeddings for CORA dataset')
    plt.show()



if __name__ == '__main__':
    dataset = datasets.Cora()
    G, node_subjects = dataset.load()
    print(G.info())

    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.1, test_size=None, stratify=node_subjects
    )

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_subjects)
    test_targets = target_encoding.transform(test_subjects)

    batch_size = 50
    num_samples = [10, 5]
    generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

    train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)

    graphsage_model = GraphSAGE(
        layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
    )

    x_inp, x_out = graphsage_model.in_out_tensors()
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    test_gen = generator.flow(test_subjects.index, test_targets)

    model.fit(train_gen, epochs=20, validation_data=test_gen, verbose=2, shuffle=False)

    model.evaluate(test_gen)

    all_nodes = node_subjects.index
    all_mapper = generator.flow(all_nodes)
    all_predictions = model.predict(all_mapper)

    node_predictions = target_encoding.inverse_transform(all_predictions)

    df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
    print(df.head(10))

    embedding_model = Model(inputs=x_inp, outputs=x_out)
    emb = embedding_model.predict(all_mapper)


    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(emb)

    plot_embeddings(X_reduced, node_subjects)

    calculate_metrics(node_subjects,node_predictions)

























