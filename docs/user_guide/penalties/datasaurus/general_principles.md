(ds-general-principles)=
# General principles for penalized synthesis

By reading through the [](ds-examples), you should hopefully gain a better understanding of what can be accomplished through the use of penalty functions with plenoptic's synthesis methods. The use of this relatively simple model and small dataset allows us to focus on the penalty functions themselves, and the following lessons should help you when developing your own penalty functions.

## Think about the construction of your penalty function

You must ensure that it encourages the property you care about while not contradicting your model. For example, let's say that your model measures the mean of its input values, so that any metamer must match the mean of your target. If your target has a mean of 0.5, then your penalty cannot constrain the range of values in the metamer to lie between -1 and 0, because there is no set of values which all lie between -1 and 0 whose mean is 0.5. See also [](how-to-penalty).

## It is generally not obvious whether your model and penalty contradict each other

Unfortunately, in general, most penalties and models are not as transparent as the example above and so it is not obvious if they contradict each other. For the examples in this section, one could work through the math by hand to see if contradictions develop, but that is much more difficult to do with, e.g., VGG16 or {class}`~plenoptic.models.PortillaSimoncelli`.

## ... but they do not need to be simultaneously satisfiable

On the other hand, while your penalty should not directly contradict your model, they do not actually need to be simultaneously satisfiable. As long as the result is a metamer and the penalty has biased optimization sufficiently so that the output has the intended characteristics, even a reasonably high penalty value is fine! In the examples in this section, none of the custom penalty functions have a near-zero value on the final metameric dataset. However, the penalties have done their job and encouraged the metameric dataset to have the intended appearance.

## {attr}`~plenoptic.Metamer.penalty_lambda` really matters

When the penalty and model are not simultaneously satisfiable, the value of {attr}`~plenoptic.Metamer.penalty_lambda` is crucial, and you will likely need to experiment with a range of values to find the value which strikes the right balance. Remember that {func}`~plenoptic.Metamer.objective_function` is literally a weighted sum of the metamer loss and penalty function, so if you double the value returned by {attr}`~plenoptic.Metamer.penalty_function`, you will likely need to halve {attr}`~plenoptic.Metamer.penalty_lambda` to see the same behavior.

## ... unless the penalty and model *are* simultaneously satisfisable

If you are lucky enough to have a penalty function and model that *can* be simultaneously satisfied, then the value of {attr}`~plenoptic.Metamer.penalty_lambda` is much less important, and a wide variety of values are likely to work. This can be seen in [plenoptic-logo](ds-plenoptic-logo), where we set the penalty function to constrain the range between 0 and 100, but do not need to change {attr}`~plenoptic.Metamer.penalty_lambda` from the default. Try changing it and compare the behavior to changing {attr}`~plenoptic.Metamer.penalty_lambda` in the other examples --- successful synthesis is much less dependent on your choice!

## Multi-stage synthesis can help when things are hard

In situations where the penalty and metamer do contradict each other, or, at least, you are unable to find a way to satisfy both reasonably well, you can do metamer synthesis in multiple stages. Those stages can consist of synthesizing the model metamer with and without the intended penalty function, and even synthesizing a metamer for some reduced model, but including the intended penalty function, and you should alternate between them. See [](ds-star) and [How to use custom penalty functions](multi-stage-penalty) for examples.
<!-- that multi-stage-penalty link *needs* title text for it to work (unlike the ds-star one, which is attached to a header) -->

## Redundant statistics change the shape of synthesis

Unrelated to the penalty function, including redundant statistics in your model can change the synthesis procedure. As discussed above, `DatasaurusModel` includes redundant statistics but, as shown in [plenoptic-logo](ds-plenoptic-logo), their presence influences the course of metamer synthesis. Different metamers are found when they're removed and, with some penalty functions, successful metamer synthesis is much more difficult without them.

## Experimentation is informative!

The process of actually developing the penalties and finding the proper configuration to synthesize metameric datasets that had the appropriate appearance was itself instructive. While, in many cases, I had a good idea how to begin formulating the penalty function, a good deal of experimentation was required in order to find a metameric dataset with the intended appearance: what value of {attr}`~plenoptic.Metamer.penalty_lambda` correctly balances the metamer loss and the penalty? what optimizer behaves best? should I tweak the penalty function? There are plenty of formulations that would all encourage the same property (e.g., minimizing the distance and the squared distance from a line both encourage the points to lie on a line), but synthesis often changes depending on which you use.

To that end, you are encouraged to experiment with the synthesis procedure shown in these notebooks. Start simple, by tweaking the tuple specifying the allowed range passed to {func}`~plenoptic.regularize.penalize_range` in the definition of `penalty` or the value of {attr}`~plenoptic.Metamer.penalty_lambda` passed to {class}`~plenoptic.Metamer`. Try changing details of the functions, e.g., by square-rooting the value returned by `circle_penalty` in [](ds_circle.md), summing the errors returned by `lines_penalty` in [](ds_hlines.md) instead of averaging them, or changing the number of points in each cluster in [](ds_polygons.md). Try changing the optimizer or its parameters (e.g., the learning rate), as shown in [plenoptic-logo](ds-plenoptic-logo). In all cases, watch the video of the synthesis process and remember: to be a metamer, the stem plots must align with the dashed horizontal lines. What effect do your changes have on the difficulty of successfully finding a metamer? On the appearance of the dataset over synthesis and in its final form? With some configurations (such as when {attr}`~plenoptic.Metamer.penalty_lambda` is too large), metamer synthesis will fail! With others, the dataset will not appear interesting! Take note of how your changes affect both of these factors and try to reason through why they had the effect they did. This problem is simpler than the models and stimuli often used with plenoptic, and thus serves as good practice.

If you create a novel interesting metamer or find an interesting synthesis process, please [let us know!](https://github.com/plenoptic-org/plenoptic/discussions/new?category=show-and-tell)
