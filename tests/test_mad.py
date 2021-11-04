# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib as mpl
# use the html backend, so we don't need to have ffmpeg
mpl.rcParams['animation.writer'] = 'html'
mpl.use('agg')
import pytest
import matplotlib.pyplot as plt
import plenoptic as po
import torch
import os.path as op
from conftest import DEVICE, DATA_DIR, get_model


class TestMAD(object):

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model', ['Identity', 'mse'], indirect=True)
    @pytest.mark.parametrize('model2', ['NLP', 'nlpd'], indirect=True)
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    def test_basic(self, curie_img, target, model, model2, store_progress, tmp_path):
        mad = po.synth.MADCompetition(curie_img, model, model2)
        mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if store_progress:
            mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)

    @pytest.mark.parametrize('loss_func', [None, 'l2', 'mse', 'range_penalty',
                                           'range_penalty_w_lmbda',
                                           'mse_range_penalty_w_lmbda'])
    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    # I think I have to do it this way (even though I'm only testing one
    # value), in order to pass the request successfully to the fixture
    @pytest.mark.parametrize('model', ['Identity'], indirect=True)
    @pytest.mark.parametrize('model2', ['NLP'], indirect=True)
    @pytest.mark.parametrize('store_progress', [False, True, 2])
    def test_loss_func(self, curie_img, model, model2, loss_func, target, store_progress, tmp_path):
        loss_kwargs = {}
        if loss_func is None:
            loss = None
        elif loss_func == 'l2':
            loss = po.tools.optim.l2_norm
        elif loss_func == 'mse':
            loss = po.tools.optim.mse
        elif loss_func == 'range_penalty':
            loss = po.tools.optim.l2_and_penalize_range
        elif loss_func == 'range_penalty_w_lmbda':
            loss = po.tools.optim.l2_and_penalize_range
            loss_kwargs['lmbda'] = .1
        elif loss_func == 'mse_range_penalty_w_lmbda':
            loss = po.tools.optim.mse_and_penalize_range
            loss_kwargs['lmbda'] = .1
        mad = po.synth.MADCompetition(curie_img, model, model2, loss_function=loss,
                                      loss_function_kwargs=loss_kwargs)
        mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                       save_progress=store_progress, save_path=op.join(tmp_path, 'test_mad.pt'))
        if store_progress:
            mad.synthesize(target, max_iter=5, loss_change_iter=3, store_progress=store_progress,
                           save_progress=store_progress,
                           save_path=op.join(tmp_path, 'test_mad.pt'), learning_rate=None,
                           initial_noise=None)

    @pytest.fixture(scope='class', params=['Identity-NLP', 'Identity-nlpd', 'mse-NLP', 'mse-nlpd'])
    def test_all(self, request, curie_img):
        # cannot parametrize within a fixture unfortunately, so we'll have to do
        # some extra instantiation here
        model, model2 = [get_model(m) for m in request.param.split('-')]
        mad = po.synth.MADCompetition(curie_img, model, model2)
        print(model2,  mad.model_1, mad.model_2, mad.loss_function_1, mad.loss_function_2)
        mad.synthesize_all(max_iter=5, loss_change_iter=3, store_progress=True,)
        print(model2,  mad.model_1, mad.model_2, mad.loss_function_1, mad.loss_function_2)
        return mad

    @pytest.mark.parametrize('resume', ['skip', 're-run', 'continue'])
    def test_all_resume(self, test_all, resume):
        test_all.synthesize_all(resume, max_iter=5, loss_change_iter=3,
                                store_progress=True, learning_rate=None,
                                initial_noise=None)

    def test_all_plot(self, test_all, tmp_path):
        test_all.plot_synthesized_image_all()
        test_all.plot_loss_all()
        plt.close('all')

    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    @pytest.mark.parametrize('model', ['SPyr', 'NLP', 'nlpd'], indirect=True)
    @pytest.mark.parametrize('model2', ['Identity'], indirect=True)
    @pytest.mark.parametrize('coarse_to_fine', ['separate', 'together'])
    def test_coarse_to_fine(self, curie_img, model, model2, target, coarse_to_fine, tmp_path):
        mad = po.synth.MADCompetition(curie_img, model, model2)
        # model needs to have the scales attribute for coarse-to-fine to work
        if hasattr(model, 'scales') and 'model_1' in target:
            mad.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                           coarse_to_fine=coarse_to_fine)
            assert len(mad.scales_finished) > 0, "Didn't actually switch scales!"
            mad.save(op.join(tmp_path, 'test_mad_ctf.pt'))
            mad_copy = po.synth.MADCompetition(curie_img, model, model2)
            mad_copy.load(op.join(tmp_path, "test_mad_ctf.pt"),
                          map_location=DEVICE)
            # check the ctf-related attributes all saved correctly
            for k in ['coarse_to_fine', 'scales', 'scales_loss', 'scales_timing',
                      'scales_finished']:
                if not getattr(mad, k) == (getattr(mad_copy, k)):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            mad_copy.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                                coarse_to_fine=coarse_to_fine)
        else:
            with pytest.raises(AttributeError):
                mad.synthesize(target, max_iter=5, loss_change_iter=1, loss_change_thresh=10,
                               coarse_to_fine=coarse_to_fine)

    @pytest.mark.parametrize('fail', [False, 'img', 'model1', 'model2', 'loss'])
    def test_save_load(self, curie_img, test_all, fail, tmp_path):
        # for ease of use
        mad = test_all
        # we need to know what model_1 and model_2 were instantiated as in
        # order to load the saved version in. if model_1 or model_2 were
        # metrics, then that object will just be an Identity class, so we need
        # to be slightly more clever about grabbing them correctly. if they're
        # metrics, they'll have a name attribute that tells us what function
        # they were. If they're models, then they won't have that attribute and
        # we just grab the object.
        try:
            model = mad.model_1.name
            if model == 'mse':
                model = po.metric.mse
            elif model == 'nlpd':
                model = po.metric.nlpd
        except AttributeError:
            model = mad.model_1
        try:
            model2 = mad.model_2.name
            if model2 == 'mse':
                model2 = po.metric.mse
            elif model2 == 'nlpd':
                model2 = po.metric.nlpd
        except AttributeError:
            model2 = mad.model_2
        mad.save(op.join(tmp_path, 'test_mad_save_load.pt'))
        # when the model is a function, the loss_function is ignored and thus
        # we won't actually fail to load here (we check against the specific
        # callable because we've overwritten the model input arg)
        if fail and not (fail == 'loss' and model == po.metric.mse and model2 == po.metric.nlpd):
            loss = None
            if fail == 'img':
                curie_img = torch.rand_like(curie_img)
            elif fail == 'model1':
                model = po.metric.nlpd
            elif fail == 'model2':
                model2 = po.metric.mse
            elif fail == 'loss':
                loss = lambda *args, **kwargs: 1
            mad_copy = po.synth.MADCompetition(curie_img, model, model2, loss_function=loss)
            with pytest.raises(Exception):
                mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"),
                              map_location=DEVICE)
        else:
            mad_copy = po.synth.MADCompetition(curie_img, model, model2)
            mad_copy.load(op.join(tmp_path, "test_mad_save_load.pt"), map_location=DEVICE)
            if mad.synthesis_target != mad_copy.synthesis_target:
                raise Exception("Something went wrong with saving and loading! synthesis_"
                                "target not the same!")
            for k in mad_copy._attrs_all:
                orig = getattr(mad, k+"_all")
                saved = getattr(mad_copy, k+"_all")
                for ki, v in orig.items():
                    eql = False
                    try:
                        if v == saved[ki]:
                            eql = True
                    except RuntimeError:
                        # then it's a tensor
                        if v.allclose(saved[ki], rtol=1E-2):
                            eql = True
                    if not eql:
                        raise Exception(f"Something went wrong with saving and loading! {k, ki} not the same!")
                eql = False
                try:
                    if saved[mad.synthesis_target] == getattr(mad_copy, k):
                        eql = True
                except RuntimeError:
                    # then it's a tensor
                    if saved[mad.synthesis_target].allclose(getattr(mad_copy, k), rtol=1E-2):
                        eql = True
                if not eql:
                    raise Exception(f"Something went wrong with saving and loading! {k} and its _all"
                                    "version not properly synced")
            # check these attributes all saved correctly
            for k in ['base_signal', 'saved_representation_1', 'saved_signal',
                      'synthesized_representation_1', 'synthesized_signal', 'base_representation_1',
                      'saved_representation_2', 'synthesized_representation_2', 'base_representation_2']:
                if not getattr(mad, k).allclose(getattr(mad_copy, k), rtol=1E-2):
                    raise Exception("Something went wrong with saving and loading! %s not the same"
                                    % k)
            assert not isinstance(mad_copy.synthesized_representation, torch.nn.Parameter), "synthesized_rep shouldn't be a parameter!"
            # check loss functions correctly saved
            mad_loss = mad.loss_function_1(mad.synthesized_representation_1, mad.base_representation_1,
                                           mad.synthesized_signal, mad.base_signal)
            mad_copy_loss = mad_copy.loss_function_1(mad_copy.synthesized_representation_1,
                                                     mad_copy.base_representation_1,
                                                     mad_copy.synthesized_signal, mad_copy.base_signal)
            if mad_loss != mad_copy_loss:
                raise Exception(f"Loss function 1 not properly saved! Before saving was {mad_loss}, "
                                f"after loading was {mad_copy_loss}")
            mad_loss = mad.loss_function_2(mad.synthesized_representation_2, mad.base_representation_2,
                                           mad.synthesized_signal, mad.base_signal)
            mad_copy_loss = mad_copy.loss_function_2(mad_copy.synthesized_representation_2,
                                                     mad_copy.base_representation_2,
                                                     mad_copy.synthesized_signal, mad_copy.base_signal)
            if mad_loss != mad_copy_loss:
                raise Exception(f"Loss function 2 not properly saved! Before saving was {mad_loss}, "
                                f"after loading was {mad_copy_loss}")
            # check that can resume
            mad_copy.synthesize('model_1_max', max_iter=5, loss_change_iter=3, store_progress=True,
                                learning_rate=None, initial_noise=None)
            # and run another synthesis target (note neither learning_rate nor
            # initial_noise can be None in this case)
            mad_copy.synthesize('model_1_min', max_iter=5, loss_change_iter=3, store_progress=True)

    @pytest.mark.parametrize('model', ['Identity', 'mse'], indirect=True)
    @pytest.mark.parametrize('model2', ['NLP', 'nlpd'], indirect=True)
    @pytest.mark.parametrize('none_arg', ['lr', 'initial_noise', 'both'])
    @pytest.mark.parametrize('none_place', ['before', 'after'])
    def test_resume_exceptions(self, curie_img, model, model2, none_arg, none_place):
        learning_rate = 1
        initial_noise = .1
        if none_arg == 'lr' or none_arg == 'both':
            learning_rate = None
        if none_arg == 'initial_noise' or none_arg == 'both':
            initial_noise = None
        mad = po.synth.MADCompetition(curie_img, model, model2)
        # can't call synthesize() with initial_noise=None or
        # learning_rate=None unless synthesize() has been called before
        # with store_progress!=False
        if none_place == 'before':
            with pytest.raises(IndexError):
                mad.synthesize('model_1_min', max_iter=5, loss_change_iter=3,
                               learning_rate=learning_rate, initial_noise=initial_noise)
        else:
            mad.synthesize('model_1_min', max_iter=5, loss_change_iter=3, store_progress=False)
            if none_arg != 'lr':
                with pytest.raises(IndexError):
                    mad.synthesize('model_1_min', max_iter=5, loss_change_iter=3,
                                   learning_rate=learning_rate, initial_noise=initial_noise)
            else:
                # this actually will not raise an exception, because the
                # learning_rate has been initialized
                mad.synthesize('model_1_min', max_iter=5, loss_change_iter=3,
                               learning_rate=learning_rate, initial_noise=initial_noise)

    @pytest.mark.parametrize('optimizer', ['Adam', 'SGD', 'Adam-args'])
    @pytest.mark.parametrize('target', ['model_1_min', 'model_2_min', 'model_1_max',
                                        'model_2_max'])
    # I think I have to do it this way (even though I'm only testing one
    # value), in order to pass the request successfully to the fixture
    @pytest.mark.parametrize('model', ['Identity'], indirect=True)
    @pytest.mark.parametrize('model2', ['NLP'], indirect=True)
    def test_optimizer_opts(self, curie_img, model, model2, optimizer,target, tmp_path):
        if '-' in optimizer:
            optimizer = optimizer.split('-')[0]
            optimizer_kwargs = {'weight_decay': .1}
        else:
            optimizer_kwargs = {}
        mad = po.synth.MADCompetition(curie_img, model, model2)
        mad.synthesize(target, max_iter=5, loss_change_iter=3,
                       optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)
